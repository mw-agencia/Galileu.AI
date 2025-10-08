using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using Galileu.Node.Core;
using Galileu.Node.Interfaces;

namespace Galileu.Node.Brain;

public class NeuralNetworkLSTM : IDisposable
{
    private readonly AdamOptimizer _adamOptimizer;
    private HybridLstmCacheManager? _cacheManager;

    public IMathTensor? weightsEmbedding { get; set; }
    public IMathTensor? weightsInputForget { get; set; }
    public IMathTensor? weightsHiddenForget { get; set; }
    public IMathTensor? weightsInputInput { get; set; }
    public IMathTensor? weightsHiddenInput { get; set; }
    public IMathTensor? weightsInputCell { get; set; }
    public IMathTensor? weightsHiddenCell { get; set; }
    public IMathTensor? weightsInputOutput { get; set; }
    public IMathTensor? weightsHiddenOutput { get; set; }
    public IMathTensor? biasForget { get; set; }
    public IMathTensor? biasInput { get; set; }
    public IMathTensor? biasCell { get; set; }
    public IMathTensor? biasOutput { get; set; }
    public IMathTensor? weightsHiddenOutputFinal { get; set; }
    public IMathTensor? biasOutputFinal { get; set; }

    protected IMathTensor? hiddenState { get; set; }
    protected IMathTensor? cellState { get; set; }

    private readonly IMathEngine _mathEngine;
    private bool _disposed = false;

    public int InputSize => inputSize;
    public int HiddenSize => hiddenSize;
    public int OutputSize => outputSize;

    private readonly int inputSize;
    private readonly int hiddenSize;
    private readonly int outputSize;

    public TensorPool? _tensorPool;

    public IMathEngine GetMathEngine() => _mathEngine;

    public NeuralNetworkLSTM(int vocabSize, int embeddingSize, int hiddenSize, int outputSize, IMathEngine mathEngine)
    {
        this.inputSize = vocabSize;
        this.hiddenSize = hiddenSize;
        this.outputSize = outputSize;
        this._mathEngine = mathEngine ?? throw new ArgumentNullException(nameof(mathEngine));
        this._adamOptimizer = new AdamOptimizer();
        
        if (_mathEngine.IsGpu)
        {
            _tensorPool = new TensorPool(_mathEngine);
        }

        hiddenState = _mathEngine.CreateTensor(new[] { 1, hiddenSize });
        cellState = _mathEngine.CreateTensor(new[] { 1, hiddenSize });

        var rand = new Random();

        weightsEmbedding = InitializeTensor(vocabSize, embeddingSize, rand);
        weightsInputForget = InitializeTensor(embeddingSize, hiddenSize, rand);
        weightsHiddenForget = InitializeTensor(hiddenSize, hiddenSize, rand);
        weightsInputInput = InitializeTensor(embeddingSize, hiddenSize, rand);
        weightsHiddenInput = InitializeTensor(hiddenSize, hiddenSize, rand);
        weightsInputCell = InitializeTensor(embeddingSize, hiddenSize, rand);
        weightsHiddenCell = InitializeTensor(hiddenSize, hiddenSize, rand);
        weightsInputOutput = InitializeTensor(embeddingSize, hiddenSize, rand);
        weightsHiddenOutput = InitializeTensor(hiddenSize, hiddenSize, rand);
        biasForget = InitializeTensor(1, hiddenSize, rand);
        biasInput = InitializeTensor(1, hiddenSize, rand);
        biasCell = InitializeTensor(1, hiddenSize, rand);
        biasOutput = InitializeTensor(1, hiddenSize, rand);
        weightsHiddenOutputFinal = InitializeTensor(hiddenSize, outputSize, rand);
        biasOutputFinal = InitializeTensor(1, outputSize, rand);
    }

    protected NeuralNetworkLSTM(
        int vocabSize, int embeddingSize, int hiddenSize, int outputSize, IMathEngine mathEngine,
        IMathTensor wEmbed, IMathTensor wif, IMathTensor whf, IMathTensor wii, IMathTensor whi,
        IMathTensor wic, IMathTensor whc, IMathTensor wio, IMathTensor who,
        IMathTensor bf, IMathTensor bi, IMathTensor bc, IMathTensor bo,
        IMathTensor why, IMathTensor by)
    {
        this.inputSize = vocabSize;
        this.hiddenSize = hiddenSize;
        this.outputSize = outputSize;
        this._mathEngine = mathEngine;
        this._adamOptimizer = new AdamOptimizer();

        if (_mathEngine.IsGpu)
        {
            _tensorPool = new TensorPool(_mathEngine);
        }

        weightsEmbedding = wEmbed;
        weightsInputForget = wif;
        weightsHiddenForget = whf;
        weightsInputInput = wii;
        weightsHiddenInput = whi;
        weightsInputCell = wic;
        weightsHiddenCell = whc;
        weightsInputOutput = wio;
        weightsHiddenOutput = who;
        biasForget = bf;
        biasInput = bi;
        biasCell = bc;
        biasOutput = bo;
        weightsHiddenOutputFinal = why;
        biasOutputFinal = by;

        hiddenState = _mathEngine.CreateTensor(new[] { 1, hiddenSize });
        cellState = _mathEngine.CreateTensor(new[] { 1, hiddenSize });
    }

    private IMathTensor InitializeTensor(int rows, int cols, Random rand)
    {
        double[] data = new double[rows * cols];
        double limit = Math.Sqrt(6.0 / (rows + cols));
        for (int i = 0; i < data.Length; i++) data[i] = (rand.NextDouble() * 2 - 1) * limit;
        return _mathEngine.CreateTensor(data, new[] { rows, cols });
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
        using var input = _mathEngine.CreateTensor(embeddedInput.GetData(), new[] { 1, embeddedInput.GetShape()[0] });

        using var fg_term1 = _mathEngine.CreateTensor(new[] { 1, hiddenSize });
        _mathEngine.MatrixMultiply(input, weightsInputForget!, fg_term1);
        using var fg_term2 = _mathEngine.CreateTensor(new[] { 1, hiddenSize });
        _mathEngine.MatrixMultiply(hiddenState!, weightsHiddenForget!, fg_term2);
        using var forgetGateLinear = _mathEngine.CreateTensor(new[] { 1, hiddenSize });
        _mathEngine.Add(fg_term1, fg_term2, forgetGateLinear);
        _mathEngine.AddBroadcast(forgetGateLinear, biasForget!, forgetGateLinear);
        using var forgetGate = _mathEngine.CreateTensor(new[] { 1, hiddenSize });
        _mathEngine.Sigmoid(forgetGateLinear, forgetGate);

        using var ig_term1 = _mathEngine.CreateTensor(new[] { 1, hiddenSize });
        _mathEngine.MatrixMultiply(input, weightsInputInput!, ig_term1);
        using var ig_term2 = _mathEngine.CreateTensor(new[] { 1, hiddenSize });
        _mathEngine.MatrixMultiply(hiddenState!, weightsHiddenInput!, ig_term2);
        using var inputGateLinear = _mathEngine.CreateTensor(new[] { 1, hiddenSize });
        _mathEngine.Add(ig_term1, ig_term2, inputGateLinear);
        _mathEngine.AddBroadcast(inputGateLinear, biasInput!, inputGateLinear);
        using var inputGate = _mathEngine.CreateTensor(new[] { 1, hiddenSize });
        _mathEngine.Sigmoid(inputGateLinear, inputGate);

        using var cc_term1 = _mathEngine.CreateTensor(new[] { 1, hiddenSize });
        _mathEngine.MatrixMultiply(input, weightsInputCell!, cc_term1);
        using var cc_term2 = _mathEngine.CreateTensor(new[] { 1, hiddenSize });
        _mathEngine.MatrixMultiply(hiddenState!, weightsHiddenCell!, cc_term2);
        using var cellCandidateLinear = _mathEngine.CreateTensor(new[] { 1, hiddenSize });
        _mathEngine.Add(cc_term1, cc_term2, cellCandidateLinear);
        _mathEngine.AddBroadcast(cellCandidateLinear, biasCell!, cellCandidateLinear);
        using var cellCandidate = _mathEngine.CreateTensor(new[] { 1, hiddenSize });
        _mathEngine.Tanh(cellCandidateLinear, cellCandidate);

        using var nextCellState_term1 = _mathEngine.CreateTensor(new[] { 1, hiddenSize });
        _mathEngine.Multiply(forgetGate, cellState!, nextCellState_term1);
        using var nextCellState_term2 = _mathEngine.CreateTensor(new[] { 1, hiddenSize });
        _mathEngine.Multiply(inputGate, cellCandidate, nextCellState_term2);
        var nextCellState = _mathEngine.CreateTensor(new[] { 1, hiddenSize });
        _mathEngine.Add(nextCellState_term1, nextCellState_term2, nextCellState);
        cellState.Dispose();
        cellState = nextCellState;

        using var og_term1 = _mathEngine.CreateTensor(new[] { 1, hiddenSize });
        _mathEngine.MatrixMultiply(input, weightsInputOutput!, og_term1);
        using var og_term2 = _mathEngine.CreateTensor(new[] { 1, hiddenSize });
        _mathEngine.MatrixMultiply(hiddenState!, weightsHiddenOutput!, og_term2);
        using var outputGateLinear = _mathEngine.CreateTensor(new[] { 1, hiddenSize });
        _mathEngine.Add(og_term1, og_term2, outputGateLinear);
        _mathEngine.AddBroadcast(outputGateLinear, biasOutput!, outputGateLinear);
        using var outputGate = _mathEngine.CreateTensor(new[] { 1, hiddenSize });
        _mathEngine.Sigmoid(outputGateLinear, outputGate);

        using var tanhCellState = _mathEngine.CreateTensor(new[] { 1, hiddenSize });
        _mathEngine.Tanh(cellState, tanhCellState);
        var nextHiddenState = _mathEngine.CreateTensor(new[] { 1, hiddenSize });
        _mathEngine.Multiply(outputGate, tanhCellState, nextHiddenState);
        hiddenState.Dispose();
        hiddenState = nextHiddenState;

        using var finalOutputLinear = _mathEngine.CreateTensor(new[] { 1, outputSize });
        _mathEngine.MatrixMultiply(hiddenState, weightsHiddenOutputFinal!, finalOutputLinear);
        _mathEngine.AddBroadcast(finalOutputLinear, biasOutputFinal!, finalOutputLinear);

        var finalOutputCpu = finalOutputLinear.ToCpuTensor();
        return new Tensor(Softmax(finalOutputCpu.GetData()), new[] { outputSize });
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

    public double TrainSequence(int[] inputIndices, Tensor[] targets, double learningRate)
    {
        _cacheManager?.Dispose();
        _cacheManager = new HybridLstmCacheManager(_mathEngine, weightsEmbedding!.Shape[1], hiddenSize, inputIndices.Length);

        using var targetsGpu = CreateSequenceTensor(targets);

        var (predictions, loss) = ForwardPassGpuOptimized(inputIndices, targetsGpu, inputIndices.Length);
        var gradients = BackwardPassGpuOptimized(targetsGpu, predictions, inputIndices, inputIndices.Length);

        UpdateWeightsWithAdamGpu(gradients, learningRate);

        predictions.Dispose();
        foreach (var grad in gradients.Values) grad.Dispose();

        return loss;
    }

    private (IMathTensor predictions, double loss) ForwardPassGpuOptimized(
        int[] inputIndices, IMathTensor targets, int sequenceLength)
    {
        var predictions = _mathEngine.CreateTensor(new[] { sequenceLength, outputSize });
        double sequenceLoss = 0;

        IMathTensor h_prev = _mathEngine.CreateTensor(new[] { 1, hiddenSize });
        IMathTensor c_prev = _mathEngine.CreateTensor(new[] { 1, hiddenSize });

        int embeddingSize = weightsEmbedding!.Shape[1];

        IMathTensor linearBuffer = _tensorPool!.Rent(new[] { 1, hiddenSize });
        IMathTensor temp1 = _tensorPool.Rent(new[] { 1, hiddenSize });
        IMathTensor temp2 = _tensorPool.Rent(new[] { 1, hiddenSize });
        IMathTensor outputLinear = _tensorPool.Rent(new[] { 1, outputSize });
        IMathTensor outputSoftmax = _tensorPool.Rent(new[] { 1, outputSize });

        try
        {
            for (int t = 0; t < sequenceLength; t++)
            {
                IMathTensor next_h, next_c;
                
                var stepCache = new LstmStepCache();
                
                stepCache.HiddenPrev = h_prev;
                stepCache.CellPrev = c_prev;

                stepCache.Input = _tensorPool.Rent(new[] { 1, embeddingSize });
                _mathEngine.Lookup(weightsEmbedding!, inputIndices[t], stepCache.Input);

                _mathEngine.MatrixMultiply(stepCache.Input, weightsInputForget!, temp1);
                _mathEngine.MatrixMultiply(h_prev, weightsHiddenForget!, temp2);
                _mathEngine.Add(temp1, temp2, linearBuffer);
                _mathEngine.AddBroadcast(linearBuffer, biasForget!, linearBuffer);
                stepCache.ForgetGate = _tensorPool.Rent(new[] { 1, hiddenSize });
                _mathEngine.Sigmoid(linearBuffer, stepCache.ForgetGate);

                _mathEngine.MatrixMultiply(stepCache.Input, weightsInputInput!, temp1);
                _mathEngine.MatrixMultiply(h_prev, weightsHiddenInput!, temp2);
                _mathEngine.Add(temp1, temp2, linearBuffer);
                _mathEngine.AddBroadcast(linearBuffer, biasInput!, linearBuffer);
                stepCache.InputGate = _tensorPool.Rent(new[] { 1, hiddenSize });
                _mathEngine.Sigmoid(linearBuffer, stepCache.InputGate);

                _mathEngine.MatrixMultiply(stepCache.Input, weightsInputCell!, temp1);
                _mathEngine.MatrixMultiply(h_prev, weightsHiddenCell!, temp2);
                _mathEngine.Add(temp1, temp2, linearBuffer);
                _mathEngine.AddBroadcast(linearBuffer, biasCell!, linearBuffer);
                stepCache.CellCandidate = _tensorPool.Rent(new[] { 1, hiddenSize });
                _mathEngine.Tanh(linearBuffer, stepCache.CellCandidate);

                stepCache.CellNext = _tensorPool.Rent(new[] { 1, hiddenSize });
                _mathEngine.Multiply(stepCache.ForgetGate, c_prev, temp1);
                _mathEngine.Multiply(stepCache.InputGate, stepCache.CellCandidate, temp2);
                _mathEngine.Add(temp1, temp2, stepCache.CellNext);

                _mathEngine.MatrixMultiply(stepCache.Input, weightsInputOutput!, temp1);
                _mathEngine.MatrixMultiply(h_prev, weightsHiddenOutput!, temp2);
                _mathEngine.Add(temp1, temp2, linearBuffer);
                _mathEngine.AddBroadcast(linearBuffer, biasOutput!, linearBuffer);
                stepCache.OutputGate = _tensorPool.Rent(new[] { 1, hiddenSize });
                _mathEngine.Sigmoid(linearBuffer, stepCache.OutputGate);

                stepCache.TanhCellNext = _tensorPool.Rent(new[] { 1, hiddenSize });
                _mathEngine.Tanh(stepCache.CellNext, stepCache.TanhCellNext);
                stepCache.HiddenNext = _tensorPool.Rent(new[] { 1, hiddenSize });
                _mathEngine.Multiply(stepCache.OutputGate, stepCache.TanhCellNext, stepCache.HiddenNext);

                _cacheManager!.CacheStep(stepCache, t);

                next_h = _mathEngine.Clone(stepCache.HiddenNext!);
                next_c = _mathEngine.Clone(stepCache.CellNext!);

                _tensorPool.Return(stepCache.Input);
                _tensorPool.Return(stepCache.ForgetGate);
                _tensorPool.Return(stepCache.InputGate);
                _tensorPool.Return(stepCache.CellCandidate);
                _tensorPool.Return(stepCache.OutputGate);
                _tensorPool.Return(stepCache.CellNext);
                _tensorPool.Return(stepCache.TanhCellNext);
                _tensorPool.Return(stepCache.HiddenNext);

                _mathEngine.MatrixMultiply(next_h, weightsHiddenOutputFinal!, outputLinear);
                _mathEngine.AddBroadcast(outputLinear, biasOutputFinal!, outputLinear);
                _mathEngine.Softmax(outputLinear, outputSoftmax);

                _mathEngine.Set(predictions, t, outputSoftmax);
                
                h_prev.Dispose();
                c_prev.Dispose();
                h_prev = next_h;
                c_prev = next_c;
            }

            var predData = predictions.ToCpuTensor().GetData();
            var targetData = targets.ToCpuTensor().GetData();
            for (int t = 0; t < sequenceLength; t++)
            {
                int offset = t * outputSize;
                int targetLocalIndex = Array.IndexOf(targetData, 1.0, offset, outputSize);
                if (targetLocalIndex != -1)
                {
                    double prob = Math.Max(predData[targetLocalIndex], 1e-9);
                    sequenceLoss += -Math.Log(prob);
                }
            }

            return (predictions, sequenceLoss / Math.Max(sequenceLength, 1));
        }
        finally
        {
            _tensorPool.Return(linearBuffer);
            _tensorPool.Return(temp1);
            _tensorPool.Return(temp2);
            _tensorPool.Return(outputLinear);
            _tensorPool.Return(outputSoftmax);
            h_prev.Dispose(); 
            c_prev.Dispose(); 
        }
    }

    private Dictionary<string, IMathTensor> BackwardPassGpuOptimized(
        IMathTensor targets, IMathTensor predictions,
        int[] inputIndices, int sequenceLength)
    {
        var grads = InitializeGradientsGpu();

        IMathTensor? dh_next = _tensorPool!.Rent(new[] { 1, hiddenSize });
        IMathTensor? dc_next = _tensorPool.Rent(new[] { 1, hiddenSize });
        IMathTensor? dy = _tensorPool.Rent(new[] { sequenceLength, outputSize });
        IMathTensor? d_embedding = _tensorPool.Rent(new[] { 1, weightsEmbedding!.Shape[1] });
        IMathTensor? current_dy = _tensorPool.Rent(new[] { 1, outputSize });
        IMathTensor? dh = _tensorPool.Rent(new[] { 1, hiddenSize });
        IMathTensor? dc = _tensorPool.Rent(new[] { 1, hiddenSize });
        IMathTensor? dog = _tensorPool.Rent(new[] { 1, hiddenSize });
        IMathTensor? dfg = _tensorPool.Rent(new[] { 1, hiddenSize });
        IMathTensor? dig = _tensorPool.Rent(new[] { 1, hiddenSize });
        IMathTensor? dcc = _tensorPool.Rent(new[] { 1, hiddenSize });
        IMathTensor? temp_deriv = _tensorPool.Rent(new[] { 1, hiddenSize });
        IMathTensor? temp_mult = _tensorPool.Rent(new[] { 1, hiddenSize });
        IMathTensor? temp_grad_why = _tensorPool.Rent(new[] { hiddenSize, outputSize });
        IMathTensor? temp_grad_w_in = _tensorPool.Rent(new[] { weightsEmbedding.Shape[1], hiddenSize });
        IMathTensor? temp_grad_w_hid = _tensorPool.Rent(new[] { hiddenSize, hiddenSize });

        try
        {
            _mathEngine.Subtract(predictions, targets, dy);

            for (int t = sequenceLength - 1; t >= 0; t--)
            {
                _mathEngine.Slice(dy, t, current_dy);
                
                using (var hiddenNext = _cacheManager!.RetrieveTensor(t, "HiddenNext"))
                {
                    _mathEngine.MatrixMultiplyTransposeA(hiddenNext, current_dy, temp_grad_why);
                    _mathEngine.Add(grads["why"], temp_grad_why, grads["why"]);
                }

                _mathEngine.Add(grads["by"], current_dy, grads["by"]);

                _mathEngine.MatrixMultiplyTransposeB(current_dy, weightsHiddenOutputFinal!, dh);
                _mathEngine.Add(dh, dh_next!, dh);

                using (var tanhCellNext = _cacheManager!.RetrieveTensor(t, "TanhCellNext"))
                using (var outputGate = _cacheManager!.RetrieveTensor(t, "OutputGate"))
                {
                    _mathEngine.Multiply(dh, tanhCellNext, dog);
                    _mathEngine.SigmoidDerivative(outputGate, temp_deriv);
                    _mathEngine.Multiply(dog, temp_deriv, dog);
                }

                using (var input = _cacheManager!.RetrieveTensor(t, "Input"))
                using (var hiddenPrev = _cacheManager!.RetrieveTensor(t, "HiddenPrev"))
                {
                    _mathEngine.MatrixMultiplyTransposeA(input, dog, temp_grad_w_in);
                    _mathEngine.Add(grads["wio"], temp_grad_w_in, grads["wio"]);
                    _mathEngine.MatrixMultiplyTransposeA(hiddenPrev, dog, temp_grad_w_hid);
                    _mathEngine.Add(grads["who"], temp_grad_w_hid, grads["who"]);
                }

                _mathEngine.Add(grads["bo"], dog, grads["bo"]);

                _mathEngine.Clone(dc_next!);
                using (var outputGate = _cacheManager!.RetrieveTensor(t, "OutputGate"))
                using (var tanhCellNext = _cacheManager!.RetrieveTensor(t, "TanhCellNext"))
                {
                    _mathEngine.Multiply(dh, outputGate, temp_mult);
                    _mathEngine.TanhDerivative(tanhCellNext, temp_deriv);
                    _mathEngine.Multiply(temp_mult, temp_deriv, temp_mult);
                    _mathEngine.Add(dc, temp_mult, dc);
                }

                using (var cellPrev = _cacheManager!.RetrieveTensor(t, "CellPrev"))
                using (var forgetGate = _cacheManager!.RetrieveTensor(t, "ForgetGate"))
                {
                    _mathEngine.Multiply(dc, cellPrev, dfg);
                    _mathEngine.SigmoidDerivative(forgetGate, temp_deriv);
                    _mathEngine.Multiply(dfg, temp_deriv, dfg);
                }
                
                using (var input = _cacheManager!.RetrieveTensor(t, "Input"))
                using (var hiddenPrev = _cacheManager!.RetrieveTensor(t, "HiddenPrev"))
                {
                    _mathEngine.MatrixMultiplyTransposeA(input, dfg, temp_grad_w_in);
                    _mathEngine.Add(grads["wif"], temp_grad_w_in, grads["wif"]);
                    _mathEngine.MatrixMultiplyTransposeA(hiddenPrev, dfg, temp_grad_w_hid);
                    _mathEngine.Add(grads["whf"], temp_grad_w_hid, grads["whf"]);
                }
                _mathEngine.Add(grads["bf"], dfg, grads["bf"]);

                using (var cellCandidate = _cacheManager!.RetrieveTensor(t, "CellCandidate"))
                using (var inputGate = _cacheManager!.RetrieveTensor(t, "InputGate"))
                {
                     _mathEngine.Multiply(dc, cellCandidate, dig);
                    _mathEngine.SigmoidDerivative(inputGate, temp_deriv);
                    _mathEngine.Multiply(dig, temp_deriv, dig);
                }
                
                using (var input = _cacheManager!.RetrieveTensor(t, "Input"))
                using (var hiddenPrev = _cacheManager!.RetrieveTensor(t, "HiddenPrev"))
                {
                     _mathEngine.MatrixMultiplyTransposeA(input, dig, temp_grad_w_in);
                    _mathEngine.Add(grads["wii"], temp_grad_w_in, grads["wii"]);
                    _mathEngine.MatrixMultiplyTransposeA(hiddenPrev, dig, temp_grad_w_hid);
                    _mathEngine.Add(grads["whi"], temp_grad_w_hid, grads["whi"]);
                }
                _mathEngine.Add(grads["bi"], dig, grads["bi"]);
                
                using (var inputGate = _cacheManager!.RetrieveTensor(t, "InputGate"))
                using (var cellCandidate = _cacheManager!.RetrieveTensor(t, "CellCandidate"))
                {
                    _mathEngine.Multiply(dc, inputGate, dcc);
                    _mathEngine.TanhDerivative(cellCandidate, temp_deriv);
                    _mathEngine.Multiply(dcc, temp_deriv, dcc);
                }
               
                using (var input = _cacheManager!.RetrieveTensor(t, "Input"))
                using (var hiddenPrev = _cacheManager!.RetrieveTensor(t, "HiddenPrev"))
                {
                     _mathEngine.MatrixMultiplyTransposeA(input, dcc, temp_grad_w_in);
                    _mathEngine.Add(grads["wic"], temp_grad_w_in, grads["wic"]);
                    _mathEngine.MatrixMultiplyTransposeA(hiddenPrev, dcc, temp_grad_w_hid);
                    _mathEngine.Add(grads["whc"], temp_grad_w_hid, grads["whc"]);
                }
                _mathEngine.Add(grads["bc"], dcc, grads["bc"]);

                _mathEngine.MatrixMultiplyTransposeB(dfg, weightsInputForget!, d_embedding);
                _mathEngine.MatrixMultiplyTransposeB(dig, weightsInputInput!, temp_mult);
                _mathEngine.Add(d_embedding, temp_mult, d_embedding);
                _mathEngine.MatrixMultiplyTransposeB(dcc, weightsInputCell!, temp_mult);
                _mathEngine.Add(d_embedding, temp_mult, d_embedding);
                _mathEngine.MatrixMultiplyTransposeB(dog, weightsInputOutput!, temp_mult);
                _mathEngine.Add(d_embedding, temp_mult, d_embedding);

                _mathEngine.AccumulateGradient(grads["w_embed"], d_embedding, inputIndices[t]);

                _mathEngine.MatrixMultiplyTransposeB(dfg, weightsHiddenForget!, dh_next!);
                _mathEngine.MatrixMultiplyTransposeB(dig, weightsHiddenInput!, temp_mult);
                _mathEngine.Add(dh_next!, temp_mult, dh_next!);
                _mathEngine.MatrixMultiplyTransposeB(dcc, weightsHiddenCell!, temp_mult);
                _mathEngine.Add(dh_next!, temp_mult, dh_next!);
                _mathEngine.MatrixMultiplyTransposeB(dog, weightsHiddenOutput!, temp_mult);
                _mathEngine.Add(dh_next!, temp_mult, dh_next!);
                
                using (var forgetGate = _cacheManager!.RetrieveTensor(t, "ForgetGate"))
                {
                    _mathEngine.Multiply(dc, forgetGate, dc_next!);
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

            return grads;
        }
        finally
        {
            _tensorPool.Return(dh_next!);
            _tensorPool.Return(dc_next!);
            _tensorPool.Return(dy);
            _tensorPool.Return(d_embedding);
            _tensorPool.Return(current_dy);
            _tensorPool.Return(dh);
            _tensorPool.Return(dc);
            _tensorPool.Return(dog);
            _tensorPool.Return(dfg);
            _tensorPool.Return(dig);
            _tensorPool.Return(dcc);
            _tensorPool.Return(temp_deriv);
            _tensorPool.Return(temp_mult);
            _tensorPool.Return(temp_grad_why);
            _tensorPool.Return(temp_grad_w_in);
            _tensorPool.Return(temp_grad_w_hid);
        }
    }

    private void UpdateWeightsWithAdamGpu(Dictionary<string, IMathTensor> gradients, double learningRate)
    {
        var parameters = new Dictionary<string, IMathTensor>
        {
            { "w_embed", weightsEmbedding! },
            { "wif", weightsInputForget! }, { "whf", weightsHiddenForget! }, { "wii", weightsInputInput! },
            { "whi", weightsHiddenInput! }, { "wic", weightsInputCell! }, { "whc", weightsHiddenCell! },
            { "wio", weightsInputOutput! }, { "who", weightsHiddenOutput! }, { "why", weightsHiddenOutputFinal! },
            { "bf", biasForget! }, { "bi", biasInput! }, { "bc", biasCell! }, { "bo", biasOutput! },
            { "by", biasOutputFinal! }
        };

        int layerId = 0;
        foreach (var key in parameters.Keys.Where(gradients.ContainsKey))
        {
            var paramTensor = parameters[key];
            var gradTensor = gradients[key];
            var paramData = paramTensor.ToCpuTensor().GetData();
            var gradData = gradTensor.ToCpuTensor().GetData();
            _adamOptimizer.UpdateParameters(layerId, paramData, gradData);
            paramTensor.UpdateFromCpu(paramData);
            layerId++;
        }
    }

    private Dictionary<string, IMathTensor> InitializeGradientsGpu()
    {
        int embeddingSize = weightsEmbedding!.Shape[1];
        return new Dictionary<string, IMathTensor>
        {
            { "w_embed", _mathEngine.CreateTensor(new[] { inputSize, embeddingSize }) },
            { "wif", _mathEngine.CreateTensor(new[] { embeddingSize, hiddenSize }) },
            { "whf", _mathEngine.CreateTensor(new[] { hiddenSize, hiddenSize }) },
            { "wii", _mathEngine.CreateTensor(new[] { embeddingSize, hiddenSize }) },
            { "whi", _mathEngine.CreateTensor(new[] { hiddenSize, hiddenSize }) },
            { "wic", _mathEngine.CreateTensor(new[] { embeddingSize, hiddenSize }) },
            { "whc", _mathEngine.CreateTensor(new[] { hiddenSize, hiddenSize }) },
            { "wio", _mathEngine.CreateTensor(new[] { embeddingSize, hiddenSize }) },
            { "who", _mathEngine.CreateTensor(new[] { hiddenSize, hiddenSize }) },
            { "why", _mathEngine.CreateTensor(new[] { hiddenSize, outputSize }) },
            { "bf", _mathEngine.CreateTensor(new[] { 1, hiddenSize }) },
            { "bi", _mathEngine.CreateTensor(new[] { 1, hiddenSize }) },
            { "bc", _mathEngine.CreateTensor(new[] { 1, hiddenSize }) },
            { "bo", _mathEngine.CreateTensor(new[] { 1, hiddenSize }) },
            { "by", _mathEngine.CreateTensor(new[] { 1, outputSize }) }
        };
    }

    private IMathTensor CreateSequenceTensor(Tensor[] sequence)
    {
        int sequenceLength = sequence.Length;
        if (sequenceLength == 0) return _mathEngine.CreateTensor(new[] { 0, 0 });
        int featureSize = sequence[0].GetShape()[0];
        var flatData = new double[sequenceLength * featureSize];
        for (int i = 0; i < sequenceLength; i++)
        {
            Array.Copy(sequence[i].GetData(), 0, flatData, i * featureSize, featureSize);
        }

        return _mathEngine.CreateTensor(flatData, new[] { sequenceLength, featureSize });
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
            weightsEmbedding?.Dispose();
            weightsInputForget?.Dispose();
            weightsHiddenForget?.Dispose();
            weightsInputInput?.Dispose();
            weightsHiddenInput?.Dispose();
            weightsInputCell?.Dispose();
            weightsHiddenCell?.Dispose();
            weightsInputOutput?.Dispose();
            weightsHiddenOutput?.Dispose();
            biasForget?.Dispose();
            biasInput?.Dispose();
            biasCell?.Dispose();
            biasOutput?.Dispose();
            weightsHiddenOutputFinal?.Dispose();
            biasOutputFinal?.Dispose();
            hiddenState?.Dispose();
            cellState?.Dispose();
            _tensorPool?.Dispose();
        }
        _disposed = true;
    }

    public void SaveModel(string filePath)
    {
        var modelData = new NeuralNetworkModelDataEmbeddingLSTM
        {
            VocabSize = this.inputSize,
            EmbeddingSize = this.weightsEmbedding!.Shape[1],
            HiddenSize = this.hiddenSize,
            OutputSize = this.outputSize,

            WeightsEmbedding = weightsEmbedding.ToCpuTensor().ToTensorData(),
            WeightsInputForget = weightsInputForget!.ToCpuTensor().ToTensorData(),
            WeightsHiddenForget = weightsHiddenForget!.ToCpuTensor().ToTensorData(),
            WeightsInputInput = weightsInputInput!.ToCpuTensor().ToTensorData(),
            WeightsHiddenInput = weightsHiddenInput!.ToCpuTensor().ToTensorData(),
            WeightsInputCell = weightsInputCell!.ToCpuTensor().ToTensorData(),
            WeightsHiddenCell = weightsHiddenCell!.ToCpuTensor().ToTensorData(),
            WeightsInputOutput = weightsInputOutput!.ToCpuTensor().ToTensorData(),
            WeightsHiddenOutput = weightsHiddenOutput!.ToCpuTensor().ToTensorData(),
            BiasForget = biasForget!.ToCpuTensor().ToTensorData(),
            BiasInput = biasInput!.ToCpuTensor().ToTensorData(),
            BiasCell = biasCell!.ToCpuTensor().ToTensorData(),
            BiasOutput = biasOutput!.ToCpuTensor().ToTensorData(),
            WeightsHiddenOutputFinal = weightsHiddenOutputFinal!.ToCpuTensor().ToTensorData(),
            BiasOutputFinal = biasOutputFinal!.ToCpuTensor().ToTensorData(),
        };
        string jsonString = JsonSerializer.Serialize(modelData, new JsonSerializerOptions { WriteIndented = true });
        File.WriteAllText(filePath, jsonString);
    }

    public static NeuralNetworkLSTM? LoadModel(string filePath, IMathEngine mathEngine)
    {
        if (!File.Exists(filePath)) return null;
        try
        {
            string jsonString = File.ReadAllText(filePath);
            var modelData = JsonSerializer.Deserialize<NeuralNetworkModelDataEmbeddingLSTM>(jsonString);
            if (modelData == null) return null;

            var wEmbed = mathEngine.CreateTensor(modelData.WeightsEmbedding.data, modelData.WeightsEmbedding.shape);
            var wif = mathEngine.CreateTensor(modelData.WeightsInputForget.data, modelData.WeightsInputForget.shape);
            var whf = mathEngine.CreateTensor(modelData.WeightsHiddenForget.data, modelData.WeightsHiddenForget.shape);
            var wii = mathEngine.CreateTensor(modelData.WeightsInputInput.data, modelData.WeightsInputInput.shape);
            var whi = mathEngine.CreateTensor(modelData.WeightsHiddenInput.data, modelData.WeightsHiddenInput.shape);
            var wic = mathEngine.CreateTensor(modelData.WeightsInputCell.data, modelData.WeightsInputCell.shape);
            var whc = mathEngine.CreateTensor(modelData.WeightsHiddenCell.data, modelData.WeightsHiddenCell.shape);
            var wio = mathEngine.CreateTensor(modelData.WeightsInputOutput.data, modelData.WeightsInputOutput.shape);
            var who = mathEngine.CreateTensor(modelData.WeightsHiddenOutput.data, modelData.WeightsHiddenOutput.shape);
            var bf = mathEngine.CreateTensor(modelData.BiasForget.data, modelData.BiasForget.shape);
            var bi = mathEngine.CreateTensor(modelData.BiasInput.data, modelData.BiasInput.shape);
            var bc = mathEngine.CreateTensor(modelData.BiasCell.data, modelData.BiasCell.shape);
            var bo = mathEngine.CreateTensor(modelData.BiasOutput.data, modelData.BiasOutput.shape);
            var why = mathEngine.CreateTensor(modelData.WeightsHiddenOutputFinal.data, modelData.WeightsHiddenOutputFinal.shape);
            var by = mathEngine.CreateTensor(modelData.BiasOutputFinal.data, modelData.BiasOutputFinal.shape);

            return new NeuralNetworkLSTM(
                modelData.VocabSize, modelData.EmbeddingSize, modelData.HiddenSize, modelData.OutputSize, mathEngine,
                wEmbed, wif, whf, wii, whi, wic, whc, wio, who, bf, bi, bc, bo, why, by
            );
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[LoadModel] Erro ao carregar o modelo: {ex.Message}");
            return null;
        }
    }

    public double CalculateSequenceLoss(int[] inputIndices, Tensor[] targets)
    {
        _cacheManager?.Dispose();
        _cacheManager = new HybridLstmCacheManager(_mathEngine, weightsEmbedding!.Shape[1], hiddenSize, inputIndices.Length);
        
        using var targetsGpu = CreateSequenceTensor(targets);
        
        var (predictions, loss) = ForwardPassGpuOptimized(inputIndices, targetsGpu, inputIndices.Length);
        
        predictions.Dispose();

        return loss;
    }
}