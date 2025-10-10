using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using Galileu.Node.Core;
using Galileu.Node.Interfaces;

namespace Galileu.Node.Brain;

/// <summary>
/// Implementação "Totalmente Out-of-Core" de uma rede neural LSTM.
/// Esta classe não armazena pesos ou buffers em RAM. Tudo é gerenciado em disco.
/// - Usa ModelParameterManager para pesos e estados do Adam.
/// - Usa StreamingLstmCacheManager para ativações (forward pass cache).
/// - Usa DiskTensorPool para todos os tensores de cálculo temporários.
/// O consumo de RAM é mínimo e constante, limitado aos tensores ativamente em uso.
/// </summary>
public class Neural : IDisposable
{
    private readonly ModelParameterManager _paramManager;
    private StreamingLstmCacheManager? _cacheManager;
    private readonly Dictionary<string, IMathTensor> _reusableGradients;

    protected IMathTensor? hiddenState { get; set; }
    protected IMathTensor? cellState { get; set; }

    private readonly IMathEngine _mathEngine;
    private bool _disposed = false;

    public int InputSize => inputSize;
    public int HiddenSize => hiddenSize;
    public int OutputSize => outputSize;
    private readonly int embeddingSize;

    private readonly int inputSize;
    private readonly int hiddenSize;
    private readonly int outputSize;
    
    public DiskTensorPool? _tensorPool;

    public IMathEngine GetMathEngine() => _mathEngine;

    public Neural(int vocabSize, int embeddingSize, int hiddenSize, int outputSize, IMathEngine mathEngine)
    {
        this.inputSize = vocabSize;
        this.hiddenSize = hiddenSize;
        this.outputSize = outputSize;
        this.embeddingSize = embeddingSize;
        this._mathEngine = mathEngine;
        
        _tensorPool = new DiskTensorPool(_mathEngine);

        var shapes = new Dictionary<string, int[]>
        {
            { "w_embed", new[] { vocabSize, embeddingSize } },
            { "wif", new[] { embeddingSize, hiddenSize } }, { "whf", new[] { hiddenSize, hiddenSize } },
            { "wii", new[] { embeddingSize, hiddenSize } }, { "whi", new[] { hiddenSize, hiddenSize } },
            { "wic", new[] { embeddingSize, hiddenSize } }, { "whc", new[] { hiddenSize, hiddenSize } },
            { "wio", new[] { embeddingSize, hiddenSize } }, { "who", new[] { hiddenSize, hiddenSize } },
            { "bf", new[] { 1, hiddenSize } }, { "bi", new[] { 1, hiddenSize } },
            { "bc", new[] { 1, hiddenSize } }, { "bo", new[] { 1, hiddenSize } },
            { "why", new[] { hiddenSize, outputSize } }, { "by", new[] { 1, outputSize } }
        };
        _paramManager = new ModelParameterManager(mathEngine, shapes);

        hiddenState = _mathEngine.CreateTensor(new[] { 1, hiddenSize });
        cellState = _mathEngine.CreateTensor(new[] { 1, hiddenSize });
        
        _reusableGradients = InitializeGradients(shapes);
    }
    
    protected Neural(string modelConfigPath, IMathEngine mathEngine)
    {
        string jsonString = File.ReadAllText(modelConfigPath);
        var modelData = JsonSerializer.Deserialize<NeuralNetworkModelDataEmbeddingLSTM>(jsonString)!;

        this.inputSize = modelData.VocabSize;
        this.hiddenSize = modelData.HiddenSize;
        this.outputSize = modelData.OutputSize;
        this.embeddingSize = modelData.EmbeddingSize;
        this._mathEngine = mathEngine;
        
        _tensorPool = new DiskTensorPool(_mathEngine);

        var shapes = new Dictionary<string, int[]>
        {
            { "w_embed", new[] { inputSize, embeddingSize } },
            { "wif", new[] { embeddingSize, hiddenSize } }, { "whf", new[] { hiddenSize, hiddenSize } },
            { "wii", new[] { embeddingSize, hiddenSize } }, { "whi", new[] { hiddenSize, hiddenSize } },
            { "wic", new[] { embeddingSize, hiddenSize } }, { "whc", new[] { hiddenSize, hiddenSize } },
            { "wio", new[] { embeddingSize, hiddenSize } }, { "who", new[] { hiddenSize, hiddenSize } },
            { "bf", new[] { 1, hiddenSize } }, { "bi", new[] { 1, hiddenSize } },
            { "bc", new[] { 1, hiddenSize } }, { "bo", new[] { 1, hiddenSize } },
            { "why", new[] { hiddenSize, outputSize } }, { "by", new[] { 1, outputSize } }
        };

        string weightsPath = Path.ChangeExtension(modelConfigPath, ".bin");
        _paramManager = new ModelParameterManager(mathEngine, weightsPath, shapes);
        
        hiddenState = _mathEngine.CreateTensor(new[] { 1, hiddenSize });
        cellState = _mathEngine.CreateTensor(new[] { 1, hiddenSize });
        
        _reusableGradients = InitializeGradients(shapes);
    }

    private Dictionary<string, IMathTensor> InitializeGradients(Dictionary<string, int[]> shapes)
    {
        var grads = new Dictionary<string, IMathTensor>();
        foreach(var pair in shapes)
        {
            grads[pair.Key] = _mathEngine.CreateTensor(pair.Value);
        }
        return grads;
    }
    
    private void ZeroOutGradients()
    {
        foreach (var grad in _reusableGradients.Values)
        {
            _mathEngine.Scale(grad, 0.0);
        }
    }

    public void ResetHiddenState()
    {
        hiddenState?.Dispose();
        cellState?.Dispose();
        hiddenState = _mathEngine.CreateTensor(new[] { 1, hiddenSize });
        cellState = _mathEngine.CreateTensor(new[] { 1, hiddenSize });
    }

    public Tensor Forward(Tensor embeddedInput)
    {
        using var input = _mathEngine.CreateTensor(embeddedInput.GetData(), embeddedInput.GetShape());

        IMathTensor forgetGate, inputGate, cellCandidate, outputGate;

        using (var wif = _paramManager.GetParameter("wif")) using (var whf = _paramManager.GetParameter("whf")) using (var bf = _paramManager.GetParameter("bf"))
        using (var fg_term1 = _mathEngine.CreateTensor(new[] { 1, hiddenSize })) using (var fg_term2 = _mathEngine.CreateTensor(new[] { 1, hiddenSize }))
        using (var forgetGateLinear = _mathEngine.CreateTensor(new[] { 1, hiddenSize }))
        {
            forgetGate = _mathEngine.CreateTensor(new[] { 1, hiddenSize });
            _mathEngine.MatrixMultiply(input, wif, fg_term1);
            _mathEngine.MatrixMultiply(hiddenState!, whf, fg_term2);
            _mathEngine.Add(fg_term1, fg_term2, forgetGateLinear);
            _mathEngine.AddBroadcast(forgetGateLinear, bf, forgetGateLinear);
            _mathEngine.Sigmoid(forgetGateLinear, forgetGate);
        }

        using (var wii = _paramManager.GetParameter("wii")) using (var whi = _paramManager.GetParameter("whi")) using (var bi = _paramManager.GetParameter("bi"))
        using (var ig_term1 = _mathEngine.CreateTensor(new[] { 1, hiddenSize })) using (var ig_term2 = _mathEngine.CreateTensor(new[] { 1, hiddenSize }))
        using (var inputGateLinear = _mathEngine.CreateTensor(new[] { 1, hiddenSize }))
        {
            inputGate = _mathEngine.CreateTensor(new[] { 1, hiddenSize });
            _mathEngine.MatrixMultiply(input, wii, ig_term1);
            _mathEngine.MatrixMultiply(hiddenState!, whi, ig_term2);
            _mathEngine.Add(ig_term1, ig_term2, inputGateLinear);
            _mathEngine.AddBroadcast(inputGateLinear, bi, inputGateLinear);
            _mathEngine.Sigmoid(inputGateLinear, inputGate);
        }

        using (var wic = _paramManager.GetParameter("wic")) using (var whc = _paramManager.GetParameter("whc")) using (var bc = _paramManager.GetParameter("bc"))
        using (var cc_term1 = _mathEngine.CreateTensor(new[] { 1, hiddenSize })) using (var cc_term2 = _mathEngine.CreateTensor(new[] { 1, hiddenSize }))
        using (var cellCandidateLinear = _mathEngine.CreateTensor(new[] { 1, hiddenSize }))
        {
            cellCandidate = _mathEngine.CreateTensor(new[] { 1, hiddenSize });
            _mathEngine.MatrixMultiply(input, wic, cc_term1);
            _mathEngine.MatrixMultiply(hiddenState!, whc, cc_term2);
            _mathEngine.Add(cc_term1, cc_term2, cellCandidateLinear);
            _mathEngine.AddBroadcast(cellCandidateLinear, bc, cellCandidateLinear);
            _mathEngine.Tanh(cellCandidateLinear, cellCandidate);
        }
        
        var nextCellState = _mathEngine.CreateTensor(new[] { 1, hiddenSize });
        using(var term1 = _mathEngine.CreateTensor(new[] { 1, hiddenSize })) using(var term2 = _mathEngine.CreateTensor(new[] { 1, hiddenSize }))
        {
            _mathEngine.Multiply(forgetGate, cellState!, term1);
            _mathEngine.Multiply(inputGate, cellCandidate, term2);
            _mathEngine.Add(term1, term2, nextCellState);
        }
        cellState.Dispose();
        cellState = nextCellState;

        using (var wio = _paramManager.GetParameter("wio")) using (var who = _paramManager.GetParameter("who")) using (var bo = _paramManager.GetParameter("bo"))
        using (var og_term1 = _mathEngine.CreateTensor(new[] { 1, hiddenSize })) using (var og_term2 = _mathEngine.CreateTensor(new[] { 1, hiddenSize }))
        using (var outputGateLinear = _mathEngine.CreateTensor(new[] { 1, hiddenSize }))
        {
            outputGate = _mathEngine.CreateTensor(new[] { 1, hiddenSize });
            _mathEngine.MatrixMultiply(input, wio, og_term1);
            _mathEngine.MatrixMultiply(hiddenState!, who, og_term2);
            _mathEngine.Add(og_term1, og_term2, outputGateLinear);
            _mathEngine.AddBroadcast(outputGateLinear, bo, outputGateLinear);
            _mathEngine.Sigmoid(outputGateLinear, outputGate);
        }
        
        var nextHiddenState = _mathEngine.CreateTensor(new[] { 1, hiddenSize });
        using(var tanhCellState = _mathEngine.CreateTensor(new[] { 1, hiddenSize }))
        {
            _mathEngine.Tanh(cellState, tanhCellState);
            _mathEngine.Multiply(outputGate, tanhCellState, nextHiddenState);
        }
        hiddenState.Dispose();
        hiddenState = nextHiddenState;

        // Liberar tensores intermediários
        forgetGate.Dispose();
        inputGate.Dispose();
        cellCandidate.Dispose();
        outputGate.Dispose();

        using (var why = _paramManager.GetParameter("why"))
        using (var by = _paramManager.GetParameter("by"))
        using (var finalOutputLinear = _mathEngine.CreateTensor(new[] { 1, outputSize }))
        {
            _mathEngine.MatrixMultiply(hiddenState!, why, finalOutputLinear);
            _mathEngine.AddBroadcast(finalOutputLinear, by, finalOutputLinear);

            var finalOutputCpu = finalOutputLinear.ToCpuTensor();
            return new Tensor(Softmax(finalOutputCpu.GetData()), new[] { outputSize });
        }
    }

    private double[] Softmax(double[] logits)
    {
        if (logits == null || logits.Length == 0) return Array.Empty<double>();
        var output = new double[logits.Length];
        double maxLogit = logits.Max();
        double sumExp = logits.Sum(l => Math.Exp(l - maxLogit));
        if (sumExp < 1e-9)
        {
            for (int i = 0; i < logits.Length; i++) output[i] = 1.0 / logits.Length;
            return output;
        }
        for (int i = 0; i < logits.Length; i++) output[i] = Math.Exp(logits[i] - maxLogit) / sumExp;
        return output;
    }
    
    public double TrainSequence(int[] inputIndices, int[] targetIndices, double learningRate)
    {
        _cacheManager?.Dispose();
        _cacheManager = new StreamingLstmCacheManager(_mathEngine, this.embeddingSize, this.hiddenSize);

        var (predictions, loss) = ForwardPassGpuOptimized(inputIndices, targetIndices, inputIndices.Length);
        
        ZeroOutGradients();
        BackwardPassGpuOptimized(predictions, inputIndices, targetIndices, inputIndices.Length, _reusableGradients);
        
        UpdateWeights(learningRate);

        predictions.Dispose();
        return loss;
    }

    private (IMathTensor predictions, double loss) ForwardPassGpuOptimized(
        int[] inputIndices, int[] targetIndices, int sequenceLength)
    {
        var predictions = _mathEngine.CreateTensor(new[] { sequenceLength, outputSize });
        double sequenceLoss = 0;

        IMathTensor h_prev = _mathEngine.CreateTensor(new[] { 1, hiddenSize });
        IMathTensor c_prev = _mathEngine.CreateTensor(new[] { 1, hiddenSize });

        using var linearBufferProxy = _tensorPool!.Rent(new[] { 1, hiddenSize }); linearBufferProxy.MarkDirty();
        using var temp1Proxy = _tensorPool.Rent(new[] { 1, hiddenSize }); temp1Proxy.MarkDirty();
        using var temp2Proxy = _tensorPool.Rent(new[] { 1, hiddenSize }); temp2Proxy.MarkDirty();
        using var outputLinearProxy = _tensorPool.Rent(new[] { 1, outputSize }); outputLinearProxy.MarkDirty();
        using var outputSoftmaxProxy = _tensorPool.Rent(new[] { 1, outputSize }); outputSoftmaxProxy.MarkDirty();

        try
        {
            for (int t = 0; t < sequenceLength; t++)
            {
                IMathTensor next_h, next_c;
                var stepCache = new LstmStepCache { HiddenPrev = h_prev, CellPrev = c_prev };

                using var inputProxy = _tensorPool.Rent(new[] { 1, embeddingSize }); inputProxy.MarkDirty();
                using var forgetGateProxy = _tensorPool.Rent(new[] { 1, hiddenSize }); forgetGateProxy.MarkDirty();
                using var inputGateProxy = _tensorPool.Rent(new[] { 1, hiddenSize }); inputGateProxy.MarkDirty();
                using var cellCandidateProxy = _tensorPool.Rent(new[] { 1, hiddenSize }); cellCandidateProxy.MarkDirty();
                using var cellNextProxy = _tensorPool.Rent(new[] { 1, hiddenSize }); cellNextProxy.MarkDirty();
                using var outputGateProxy = _tensorPool.Rent(new[] { 1, hiddenSize }); outputGateProxy.MarkDirty();
                using var tanhCellNextProxy = _tensorPool.Rent(new[] { 1, hiddenSize }); tanhCellNextProxy.MarkDirty();
                using var hiddenNextProxy = _tensorPool.Rent(new[] { 1, hiddenSize }); hiddenNextProxy.MarkDirty();
                
                using (var w_embed = _paramManager.GetParameter("w_embed")) _mathEngine.Lookup(w_embed, inputIndices[t], inputProxy.GetTensor());
                stepCache.Input = inputProxy.GetTensor();

                using (var wif = _paramManager.GetParameter("wif")) using (var whf = _paramManager.GetParameter("whf")) using (var bf = _paramManager.GetParameter("bf"))
                {
                    _mathEngine.MatrixMultiply(inputProxy.GetTensor(), wif, temp1Proxy.GetTensor());
                    _mathEngine.MatrixMultiply(h_prev, whf, temp2Proxy.GetTensor());
                    _mathEngine.Add(temp1Proxy.GetTensor(), temp2Proxy.GetTensor(), linearBufferProxy.GetTensor());
                    _mathEngine.AddBroadcast(linearBufferProxy.GetTensor(), bf, linearBufferProxy.GetTensor());
                    _mathEngine.Sigmoid(linearBufferProxy.GetTensor(), forgetGateProxy.GetTensor());
                    stepCache.ForgetGate = forgetGateProxy.GetTensor();
                }
                
                using (var wii = _paramManager.GetParameter("wii")) using (var whi = _paramManager.GetParameter("whi")) using (var bi = _paramManager.GetParameter("bi"))
                {
                    _mathEngine.MatrixMultiply(inputProxy.GetTensor(), wii, temp1Proxy.GetTensor());
                    _mathEngine.MatrixMultiply(h_prev, whi, temp2Proxy.GetTensor());
                    _mathEngine.Add(temp1Proxy.GetTensor(), temp2Proxy.GetTensor(), linearBufferProxy.GetTensor());
                    _mathEngine.AddBroadcast(linearBufferProxy.GetTensor(), bi, linearBufferProxy.GetTensor());
                    _mathEngine.Sigmoid(linearBufferProxy.GetTensor(), inputGateProxy.GetTensor());
                    stepCache.InputGate = inputGateProxy.GetTensor();
                }

                using (var wic = _paramManager.GetParameter("wic")) using (var whc = _paramManager.GetParameter("whc")) using (var bc = _paramManager.GetParameter("bc"))
                {
                    _mathEngine.MatrixMultiply(inputProxy.GetTensor(), wic, temp1Proxy.GetTensor());
                    _mathEngine.MatrixMultiply(h_prev, whc, temp2Proxy.GetTensor());
                    _mathEngine.Add(temp1Proxy.GetTensor(), temp2Proxy.GetTensor(), linearBufferProxy.GetTensor());
                    _mathEngine.AddBroadcast(linearBufferProxy.GetTensor(), bc, linearBufferProxy.GetTensor());
                    _mathEngine.Tanh(linearBufferProxy.GetTensor(), cellCandidateProxy.GetTensor());
                    stepCache.CellCandidate = cellCandidateProxy.GetTensor();
                }

                _mathEngine.Multiply(forgetGateProxy.GetTensor(), c_prev, temp1Proxy.GetTensor());
                _mathEngine.Multiply(inputGateProxy.GetTensor(), cellCandidateProxy.GetTensor(), temp2Proxy.GetTensor());
                _mathEngine.Add(temp1Proxy.GetTensor(), temp2Proxy.GetTensor(), cellNextProxy.GetTensor());
                stepCache.CellNext = cellNextProxy.GetTensor();

                using (var wio = _paramManager.GetParameter("wio")) using (var who = _paramManager.GetParameter("who")) using (var bo = _paramManager.GetParameter("bo"))
                {
                    _mathEngine.MatrixMultiply(inputProxy.GetTensor(), wio, temp1Proxy.GetTensor());
                    _mathEngine.MatrixMultiply(h_prev, who, temp2Proxy.GetTensor());
                    _mathEngine.Add(temp1Proxy.GetTensor(), temp2Proxy.GetTensor(), linearBufferProxy.GetTensor());
                    _mathEngine.AddBroadcast(linearBufferProxy.GetTensor(), bo, linearBufferProxy.GetTensor());
                    _mathEngine.Sigmoid(linearBufferProxy.GetTensor(), outputGateProxy.GetTensor());
                    stepCache.OutputGate = outputGateProxy.GetTensor();
                }

                _mathEngine.Tanh(cellNextProxy.GetTensor(), tanhCellNextProxy.GetTensor());
                stepCache.TanhCellNext = tanhCellNextProxy.GetTensor();
                _mathEngine.Multiply(outputGateProxy.GetTensor(), tanhCellNextProxy.GetTensor(), hiddenNextProxy.GetTensor());
                stepCache.HiddenNext = hiddenNextProxy.GetTensor();

                _cacheManager!.CacheStep(stepCache, t);

                next_h = _mathEngine.Clone(stepCache.HiddenNext!);
                next_c = _mathEngine.Clone(stepCache.CellNext!);

                using (var why = _paramManager.GetParameter("why")) using (var by = _paramManager.GetParameter("by"))
                {
                    _mathEngine.MatrixMultiply(next_h, why, outputLinearProxy.GetTensor());
                    _mathEngine.AddBroadcast(outputLinearProxy.GetTensor(), by, outputLinearProxy.GetTensor());
                    _mathEngine.Softmax(outputLinearProxy.GetTensor(), outputSoftmaxProxy.GetTensor());
                }

                _mathEngine.Set(predictions, t, outputSoftmaxProxy.GetTensor());
                
                h_prev.Dispose();
                c_prev.Dispose();
                h_prev = next_h;
                c_prev = next_c;
            }

            var predData = predictions.ToCpuTensor().GetData();
            for (int t = 0; t < sequenceLength; t++)
            {
                int targetIndex = targetIndices[t];
                int flatIndex = t * outputSize + targetIndex;
                double prob = Math.Max(predData[flatIndex], 1e-9);
                sequenceLoss += -Math.Log(prob);
            }

            return (predictions, sequenceLoss / Math.Max(sequenceLength, 1));
        }
        finally
        {
            h_prev.Dispose(); 
            c_prev.Dispose(); 
        }
    }

    private void BackwardPassGpuOptimized(
        IMathTensor predictions, int[] inputIndices, int[] targetIndices, int sequenceLength, Dictionary<string, IMathTensor> grads)
    {
        using var dh_nextProxy = _tensorPool!.Rent(new[] { 1, hiddenSize }); dh_nextProxy.MarkDirty();
        using var dc_nextProxy = _tensorPool.Rent(new[] { 1, hiddenSize }); dc_nextProxy.MarkDirty();
        using var dyProxy = _tensorPool.Rent(new[] { sequenceLength, outputSize }); dyProxy.MarkDirty();
        using var d_embeddingProxy = _tensorPool.Rent(new[] { 1, embeddingSize }); d_embeddingProxy.MarkDirty();
        using var current_dyProxy = _tensorPool.Rent(new[] { 1, outputSize }); current_dyProxy.MarkDirty();
        using var dhProxy = _tensorPool.Rent(new[] { 1, hiddenSize }); dhProxy.MarkDirty();
        using var dcProxy = _tensorPool.Rent(new[] { 1, hiddenSize }); dcProxy.MarkDirty();
        using var dogProxy = _tensorPool.Rent(new[] { 1, hiddenSize }); dogProxy.MarkDirty();
        using var dfgProxy = _tensorPool.Rent(new[] { 1, hiddenSize }); dfgProxy.MarkDirty();
        using var digProxy = _tensorPool.Rent(new[] { 1, hiddenSize }); digProxy.MarkDirty();
        using var dccProxy = _tensorPool.Rent(new[] { 1, hiddenSize }); dccProxy.MarkDirty();
        using var temp_derivProxy = _tensorPool.Rent(new[] { 1, hiddenSize }); temp_derivProxy.MarkDirty();
        using var temp_multProxy = _tensorPool.Rent(new[] { 1, hiddenSize }); temp_multProxy.MarkDirty();
        using var temp_grad_w_inProxy = _tensorPool.Rent(new[] { embeddingSize, hiddenSize }); temp_grad_w_inProxy.MarkDirty();
        using var temp_grad_w_hidProxy = _tensorPool.Rent(new[] { hiddenSize, hiddenSize }); temp_grad_w_hidProxy.MarkDirty();

        try
        {
            _mathEngine.SoftmaxCrossEntropyGradient(predictions, targetIndices, dyProxy.GetTensor());

            for (int t = sequenceLength - 1; t >= 0; t--)
            {
                _mathEngine.Slice(dyProxy.GetTensor(), t, current_dyProxy.GetTensor());
                
                using (var hiddenNext = _cacheManager!.RetrieveTensor(t, "HiddenNext"))
                using (var temp_grad_whyProxy = _tensorPool.Rent(new[] { hiddenSize, outputSize }))
                { 
                    temp_grad_whyProxy.MarkDirty();
                    _mathEngine.MatrixMultiplyTransposeA(hiddenNext, current_dyProxy.GetTensor(), temp_grad_whyProxy.GetTensor());
                    _mathEngine.Add(grads["why"], temp_grad_whyProxy.GetTensor(), grads["why"]);
                }

                _mathEngine.Add(grads["by"], current_dyProxy.GetTensor(), grads["by"]);

                using (var why = _paramManager.GetParameter("why"))
                {
                    _mathEngine.MatrixMultiplyTransposeB(current_dyProxy.GetTensor(), why, dhProxy.GetTensor());
                }
                _mathEngine.Add(dhProxy.GetTensor(), dh_nextProxy.GetTensor(), dh_nextProxy.GetTensor());

                using (var tanhCellNext = _cacheManager!.RetrieveTensor(t, "TanhCellNext"))
                using (var outputGate = _cacheManager!.RetrieveTensor(t, "OutputGate"))
                {
                    _mathEngine.Multiply(dh_nextProxy.GetTensor(), tanhCellNext, dogProxy.GetTensor());
                    _mathEngine.SigmoidDerivative(outputGate, temp_derivProxy.GetTensor());
                    _mathEngine.Multiply(dogProxy.GetTensor(), temp_derivProxy.GetTensor(), dogProxy.GetTensor());
                }

                using (var input = _cacheManager!.RetrieveTensor(t, "Input"))
                using (var hiddenPrev = _cacheManager!.RetrieveTensor(t, "HiddenPrev"))
                {
                    _mathEngine.MatrixMultiplyTransposeA(input, dogProxy.GetTensor(), temp_grad_w_inProxy.GetTensor());
                    _mathEngine.Add(grads["wio"], temp_grad_w_inProxy.GetTensor(), grads["wio"]);
                    _mathEngine.MatrixMultiplyTransposeA(hiddenPrev, dogProxy.GetTensor(), temp_grad_w_hidProxy.GetTensor());
                    _mathEngine.Add(grads["who"], temp_grad_w_hidProxy.GetTensor(), grads["who"]);
                }

                _mathEngine.Add(grads["bo"], dogProxy.GetTensor(), grads["bo"]);

                _mathEngine.Copy(dc_nextProxy.GetTensor(), dcProxy.GetTensor());
                using (var outputGate = _cacheManager!.RetrieveTensor(t, "OutputGate"))
                using (var tanhCellNext = _cacheManager!.RetrieveTensor(t, "TanhCellNext"))
                {
                    _mathEngine.Multiply(dh_nextProxy.GetTensor(), outputGate, temp_multProxy.GetTensor());
                    _mathEngine.TanhDerivative(tanhCellNext, temp_derivProxy.GetTensor());
                    _mathEngine.Multiply(temp_multProxy.GetTensor(), temp_derivProxy.GetTensor(), temp_multProxy.GetTensor());
                    _mathEngine.Add(dcProxy.GetTensor(), temp_multProxy.GetTensor(), dcProxy.GetTensor());
                }

                using (var cellPrev = _cacheManager!.RetrieveTensor(t, "CellPrev"))
                using (var forgetGate = _cacheManager!.RetrieveTensor(t, "ForgetGate"))
                {
                    _mathEngine.Multiply(dcProxy.GetTensor(), cellPrev, dfgProxy.GetTensor());
                    _mathEngine.SigmoidDerivative(forgetGate, temp_derivProxy.GetTensor());
                    _mathEngine.Multiply(dfgProxy.GetTensor(), temp_derivProxy.GetTensor(), dfgProxy.GetTensor());
                }
                
                using (var input = _cacheManager!.RetrieveTensor(t, "Input"))
                using (var hiddenPrev = _cacheManager!.RetrieveTensor(t, "HiddenPrev"))
                {
                    _mathEngine.MatrixMultiplyTransposeA(input, dfgProxy.GetTensor(), temp_grad_w_inProxy.GetTensor());
                    _mathEngine.Add(grads["wif"], temp_grad_w_inProxy.GetTensor(), grads["wif"]);
                    _mathEngine.MatrixMultiplyTransposeA(hiddenPrev, dfgProxy.GetTensor(), temp_grad_w_hidProxy.GetTensor());
                    _mathEngine.Add(grads["whf"], temp_grad_w_hidProxy.GetTensor(), grads["whf"]);
                }
                _mathEngine.Add(grads["bf"], dfgProxy.GetTensor(), grads["bf"]);

                using (var cellCandidate = _cacheManager!.RetrieveTensor(t, "CellCandidate"))
                using (var inputGate = _cacheManager!.RetrieveTensor(t, "InputGate"))
                {
                     _mathEngine.Multiply(dcProxy.GetTensor(), cellCandidate, digProxy.GetTensor());
                    _mathEngine.SigmoidDerivative(inputGate, temp_derivProxy.GetTensor());
                    _mathEngine.Multiply(digProxy.GetTensor(), temp_derivProxy.GetTensor(), digProxy.GetTensor());
                }
                
                using (var input = _cacheManager!.RetrieveTensor(t, "Input"))
                using (var hiddenPrev = _cacheManager!.RetrieveTensor(t, "HiddenPrev"))
                {
                     _mathEngine.MatrixMultiplyTransposeA(input, digProxy.GetTensor(), temp_grad_w_inProxy.GetTensor());
                    _mathEngine.Add(grads["wii"], temp_grad_w_inProxy.GetTensor(), grads["wii"]);
                    _mathEngine.MatrixMultiplyTransposeA(hiddenPrev, digProxy.GetTensor(), temp_grad_w_hidProxy.GetTensor());
                    _mathEngine.Add(grads["whi"], temp_grad_w_hidProxy.GetTensor(), grads["whi"]);
                }
                _mathEngine.Add(grads["bi"], digProxy.GetTensor(), grads["bi"]);
                
                using (var inputGate = _cacheManager!.RetrieveTensor(t, "InputGate"))
                using (var cellCandidate = _cacheManager!.RetrieveTensor(t, "CellCandidate"))
                {
                    _mathEngine.Multiply(dcProxy.GetTensor(), inputGate, dccProxy.GetTensor());
                    _mathEngine.TanhDerivative(cellCandidate, temp_derivProxy.GetTensor());
                    _mathEngine.Multiply(dccProxy.GetTensor(), temp_derivProxy.GetTensor(), dccProxy.GetTensor());
                }
               
                using (var input = _cacheManager!.RetrieveTensor(t, "Input"))
                using (var hiddenPrev = _cacheManager!.RetrieveTensor(t, "HiddenPrev"))
                {
                     _mathEngine.MatrixMultiplyTransposeA(input, dccProxy.GetTensor(), temp_grad_w_inProxy.GetTensor());
                    _mathEngine.Add(grads["wic"], temp_grad_w_inProxy.GetTensor(), grads["wic"]);
                    _mathEngine.MatrixMultiplyTransposeA(hiddenPrev, dccProxy.GetTensor(), temp_grad_w_hidProxy.GetTensor());
                    _mathEngine.Add(grads["whc"], temp_grad_w_hidProxy.GetTensor(), grads["whc"]);
                }
                _mathEngine.Add(grads["bc"], dccProxy.GetTensor(), grads["bc"]);

                using(var wif = _paramManager.GetParameter("wif")) _mathEngine.MatrixMultiplyTransposeB(dfgProxy.GetTensor(), wif, d_embeddingProxy.GetTensor());
                using(var wii = _paramManager.GetParameter("wii")) { _mathEngine.MatrixMultiplyTransposeB(digProxy.GetTensor(), wii, temp_multProxy.GetTensor()); _mathEngine.Add(d_embeddingProxy.GetTensor(), temp_multProxy.GetTensor(), d_embeddingProxy.GetTensor()); }
                using(var wic = _paramManager.GetParameter("wic")) { _mathEngine.MatrixMultiplyTransposeB(dccProxy.GetTensor(), wic, temp_multProxy.GetTensor()); _mathEngine.Add(d_embeddingProxy.GetTensor(), temp_multProxy.GetTensor(), d_embeddingProxy.GetTensor()); }
                using(var wio = _paramManager.GetParameter("wio")) { _mathEngine.MatrixMultiplyTransposeB(dogProxy.GetTensor(), wio, temp_multProxy.GetTensor()); _mathEngine.Add(d_embeddingProxy.GetTensor(), temp_multProxy.GetTensor(), d_embeddingProxy.GetTensor()); }

                _mathEngine.AccumulateGradient(grads["w_embed"], d_embeddingProxy.GetTensor(), inputIndices[t]);

                using(var whf = _paramManager.GetParameter("whf")) _mathEngine.MatrixMultiplyTransposeB(dfgProxy.GetTensor(), whf, dh_nextProxy.GetTensor());
                using(var whi = _paramManager.GetParameter("whi")) { _mathEngine.MatrixMultiplyTransposeB(digProxy.GetTensor(), whi, temp_multProxy.GetTensor()); _mathEngine.Add(dh_nextProxy.GetTensor(), temp_multProxy.GetTensor(), dh_nextProxy.GetTensor()); }
                using(var whc = _paramManager.GetParameter("whc")) { _mathEngine.MatrixMultiplyTransposeB(dccProxy.GetTensor(), whc, temp_multProxy.GetTensor()); _mathEngine.Add(dh_nextProxy.GetTensor(), temp_multProxy.GetTensor(), dh_nextProxy.GetTensor()); }
                using(var who = _paramManager.GetParameter("who")) { _mathEngine.MatrixMultiplyTransposeB(dogProxy.GetTensor(), who, temp_multProxy.GetTensor()); _mathEngine.Add(dh_nextProxy.GetTensor(), temp_multProxy.GetTensor(), dh_nextProxy.GetTensor()); }
                
                using (var forgetGate = _cacheManager!.RetrieveTensor(t, "ForgetGate"))
                {
                    _mathEngine.Multiply(dcProxy.GetTensor(), forgetGate, dc_nextProxy.GetTensor());
                }
            }

            double totalNorm = 0;
            foreach (var grad in grads.Values)
            {
                totalNorm += grad.ToCpuTensor().GetData().Sum(x => x * x);
            }

            totalNorm = Math.Sqrt(totalNorm);
            const double MAX_GRAD_NORM = 5.0;
            if (totalNorm > MAX_GRAD_NORM)
            {
                double clipScale = MAX_GRAD_NORM / (totalNorm + 1e-8);
                foreach (var grad in grads.Values)
                {
                    _mathEngine.Scale(grad, clipScale);
                }
            }
        }
        finally
        {
            // O bloco 'using' no início do método garante o descarte automático de todos os proxies.
        }
    }

    private void UpdateWeights(double learningRate)
    {
        foreach (var name in _reusableGradients.Keys)
        {
            _paramManager.UpdateParameter(name, _reusableGradients[name], learningRate);
        }
    }

    public void Dispose()
    {
        Dispose(true);
        GC.SuppressFinalize(this);
    }

    protected virtual void Dispose(bool disposing)
    {
        if (_disposed) return;
        if (disposing)
        {
            _cacheManager?.Dispose();
            _paramManager?.Dispose();
            hiddenState?.Dispose();
            cellState?.Dispose();
            _tensorPool?.Dispose();

            if (_reusableGradients != null)
            {
                foreach(var grad in _reusableGradients.Values)
                {
                    grad.Dispose();
                }
                _reusableGradients.Clear();
            }
        }
        _disposed = true;
    }

    public void SaveModel(string filePath)
    {
        _paramManager.SaveToFile(Path.ChangeExtension(filePath, ".bin"));
        
        var modelData = new NeuralNetworkModelDataEmbeddingLSTM
        {
            VocabSize = this.inputSize,
            EmbeddingSize = this.embeddingSize,
            HiddenSize = this.hiddenSize,
            OutputSize = this.outputSize
        };
        string jsonString = JsonSerializer.Serialize(modelData, new JsonSerializerOptions { WriteIndented = true });
        File.WriteAllText(filePath, jsonString);
    }

    public static Neural? LoadModel(string modelConfigPath, IMathEngine mathEngine)
    {
        if (!File.Exists(modelConfigPath)) return null;

        string weightsPath = Path.ChangeExtension(modelConfigPath, ".bin");
        if (!File.Exists(weightsPath)) return null;

        return new Neural(modelConfigPath, mathEngine);
    }

    public double CalculateSequenceLoss(int[] inputIndices, int[] targetIndices)
    {
        _cacheManager?.Dispose();
        _cacheManager = new StreamingLstmCacheManager(_mathEngine, this.embeddingSize, this.hiddenSize);
        
        var (predictions, loss) = ForwardPassGpuOptimized(inputIndices, targetIndices, inputIndices.Length);
        
        predictions.Dispose();
        return loss;
    }
}