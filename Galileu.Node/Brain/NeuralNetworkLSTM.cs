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

    // --- CORREÇÃO 1: Usando o gerenciador de cache correto e mais eficiente ---
    private readonly DiskOnlyCacheManager _cacheManager;

    // Pesos e biases do modelo
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

        // --- CORREÇÃO 1: Instanciando o gerenciador de cache correto ---
        _cacheManager = new DiskOnlyCacheManager(_mathEngine, embeddingSize, hiddenSize);
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

        // --- CORREÇÃO 1: Instanciando o gerenciador de cache correto ---
        _cacheManager = new DiskOnlyCacheManager(_mathEngine, embeddingSize, hiddenSize);
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

    // ========================================
// TRAIN SEQUENCE - FIX CRÍTICO DE GRADIENTES
// Garante liberação de TODOS os tensores
// ========================================

    public double TrainSequence(int[] inputIndices, Tensor[] targets, double learningRate)
    {
        // ✅ Variáveis nullable para garantir limpeza no finally
        IMathTensor? targetsGpu = null;
        IMathTensor? predictions = null;
        Dictionary<string, IMathTensor>? gradients = null;

        try
        {
            // ✅ FASE 1: Reset do cache (trunca arquivo de disco)
            _cacheManager.Reset();

            // ✅ FASE 2: Converte targets para GPU
            targetsGpu = CreateSequenceTensor(targets);

            // ✅ FASE 3: Forward Pass
            var (pred, loss) = ForwardPassGpuOptimized(inputIndices, targetsGpu, inputIndices.Length);
            predictions = pred;

            // ✅ FASE 4: Backward Pass (cria 15 tensores de gradientes)
            gradients = BackwardPassGpuOptimized(targetsGpu, predictions, inputIndices, inputIndices.Length);

            // ✅ FASE 5: Update Weights (usa os gradientes)
            UpdateWeightsWithAdamGpu(gradients, learningRate);

            // ✅✅✅ FASE 6: LIBERA GRADIENTES (CRÍTICO!)
            // Este é o vazamento que você identificou corretamente!
            foreach (var grad in gradients.Values)
            {
                grad?.Dispose();
            }

            gradients.Clear();
            gradients = null; // Marca como null para evitar double-dispose no finally

            // ✅ FASE 7: Libera predictions
            predictions?.Dispose();
            predictions = null;

            // ✅ FASE 8: Libera targets GPU
            targetsGpu?.Dispose();
            targetsGpu = null;

            // ✅✅✅ FASE 9: LIMPEZA ULTRA-AGRESSIVA
            // Força sincronização da GPU (evita comandos pendentes)
            if (_mathEngine.IsGpu)
            {
                var gpuEngine = _mathEngine as Galileu.Node.Gpu.GpuMathEngine;
                gpuEngine?.Synchronize();
            }

            // ✅ Trim do TensorPool após cada batch
            _tensorPool?.Trim();

            // ✅ Sugere GC leve (Gen 0 apenas, rápido)
            GC.Collect(0, GCCollectionMode.Optimized, false);

            return loss;
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[ERRO] TrainSequence falhou: {ex.Message}");
            throw;
        }
        finally
        {
            // ✅✅✅ GARANTIA FINAL: Libera TUDO mesmo se houver exceção

            // Libera predictions
            if (predictions != null)
            {
                predictions.Dispose();
                predictions = null;
            }

            // Libera targets GPU
            if (targetsGpu != null)
            {
                targetsGpu.Dispose();
                targetsGpu = null;
            }

            // ✅✅✅ CRÍTICO: Libera gradientes
            if (gradients != null)
            {
                foreach (var grad in gradients.Values)
                {
                    grad?.Dispose();
                }

                gradients.Clear();
                gradients = null;
            }
        }
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
                IMathTensor inputEmbedding = _tensorPool.Rent(new[] { 1, embeddingSize });
                IMathTensor forgetGate = _tensorPool.Rent(new[] { 1, hiddenSize });
                IMathTensor inputGate = _tensorPool.Rent(new[] { 1, hiddenSize });
                IMathTensor cellCandidate = _tensorPool.Rent(new[] { 1, hiddenSize });
                IMathTensor outputGate = _tensorPool.Rent(new[] { 1, hiddenSize });
                IMathTensor cellNext = _tensorPool.Rent(new[] { 1, hiddenSize });
                IMathTensor tanhCellNext = _tensorPool.Rent(new[] { 1, hiddenSize });
                IMathTensor hiddenNext = _tensorPool.Rent(new[] { 1, hiddenSize });

                _mathEngine.Lookup(weightsEmbedding!, inputIndices[t], inputEmbedding);

                // Forget Gate
                _mathEngine.MatrixMultiply(inputEmbedding, weightsInputForget!, temp1);
                _mathEngine.MatrixMultiply(h_prev, weightsHiddenForget!, temp2);
                _mathEngine.Add(temp1, temp2, linearBuffer);
                _mathEngine.AddBroadcast(linearBuffer, biasForget!, linearBuffer);
                _mathEngine.Sigmoid(linearBuffer, forgetGate);

                // Input Gate
                _mathEngine.MatrixMultiply(inputEmbedding, weightsInputInput!, temp1);
                _mathEngine.MatrixMultiply(h_prev, weightsHiddenInput!, temp2);
                _mathEngine.Add(temp1, temp2, linearBuffer);
                _mathEngine.AddBroadcast(linearBuffer, biasInput!, linearBuffer);
                _mathEngine.Sigmoid(linearBuffer, inputGate);

                // Cell Candidate
                _mathEngine.MatrixMultiply(inputEmbedding, weightsInputCell!, temp1);
                _mathEngine.MatrixMultiply(h_prev, weightsHiddenCell!, temp2);
                _mathEngine.Add(temp1, temp2, linearBuffer);
                _mathEngine.AddBroadcast(linearBuffer, biasCell!, linearBuffer);
                _mathEngine.Tanh(linearBuffer, cellCandidate);

                // Cell State
                _mathEngine.Multiply(forgetGate, c_prev, temp1);
                _mathEngine.Multiply(inputGate, cellCandidate, temp2);
                _mathEngine.Add(temp1, temp2, cellNext);

                // Output Gate
                _mathEngine.MatrixMultiply(inputEmbedding, weightsInputOutput!, temp1);
                _mathEngine.MatrixMultiply(h_prev, weightsHiddenOutput!, temp2);
                _mathEngine.Add(temp1, temp2, linearBuffer);
                _mathEngine.AddBroadcast(linearBuffer, biasOutput!, linearBuffer);
                _mathEngine.Sigmoid(linearBuffer, outputGate);

                // Hidden State
                _mathEngine.Tanh(cellNext, tanhCellNext);
                _mathEngine.Multiply(outputGate, tanhCellNext, hiddenNext);

                IMathTensor cloneHiddenPrev = _mathEngine.Clone(h_prev);
                IMathTensor cloneCellPrev = _mathEngine.Clone(c_prev);

                var stepCache = new LstmStepCache
                {
                    Input = inputEmbedding, HiddenPrev = cloneHiddenPrev, CellPrev = cloneCellPrev,
                    ForgetGate = forgetGate, InputGate = inputGate, CellCandidate = cellCandidate,
                    OutputGate = outputGate, CellNext = cellNext, TanhCellNext = tanhCellNext,
                    HiddenNext = hiddenNext
                };

                _cacheManager.CacheStep(stepCache);

                _mathEngine.MatrixMultiply(hiddenNext, weightsHiddenOutputFinal!, outputLinear);
                _mathEngine.AddBroadcast(outputLinear, biasOutputFinal!, outputLinear);
                _mathEngine.Softmax(outputLinear, outputSoftmax);
                _mathEngine.Set(predictions, t, outputSoftmax);

                h_prev.Dispose();
                c_prev.Dispose();

                h_prev = _mathEngine.Clone(hiddenNext);
                c_prev = _mathEngine.Clone(cellNext);
            }

            var predData = predictions.ToCpuTensor().GetData();
            var targetData = targets.ToCpuTensor().GetData();

            for (int t = 0; t < sequenceLength; t++)
            {
                int offset = t * outputSize;
                int targetLocalIndex = Array.IndexOf(targetData, 1.0, offset, outputSize);
                if (targetLocalIndex != -1)
                {
                    double prob = Math.Max(predData[offset + targetLocalIndex], 1e-9);
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
            h_prev?.Dispose();
            c_prev?.Dispose();
        }
    }

    // ========================================
// BACKWARD PASS - LIMPEZA TOTAL
// Libera TODOS os tensores incluindo gradientes
// ========================================

    private Dictionary<string, IMathTensor> BackwardPassGpuOptimized(
        IMathTensor targets, IMathTensor predictions,
        int[] inputIndices, int sequenceLength)
    {
        // ✅ Inicializa gradientes (15 tensores criados aqui)
        var grads = InitializeGradientsGpu();

        // ✅ Buffers temporários (nullable para garantir limpeza)
        IMathTensor? dh_next = null;
        IMathTensor? dc_next = null;
        IMathTensor? dy = null;
        IMathTensor? d_embedding = null;
        IMathTensor? current_dy = null;
        IMathTensor? dh = null;
        IMathTensor? dc = null;
        IMathTensor? dog = null;
        IMathTensor? dfg = null;
        IMathTensor? dig = null;
        IMathTensor? dcc = null;
        IMathTensor? temp_deriv = null;
        IMathTensor? temp_mult = null;
        IMathTensor? temp_grad_why = null;
        IMathTensor? temp_grad_w_in = null;
        IMathTensor? temp_grad_w_hid = null;

        try
        {
            // ✅ FASE 1: Aloca TODOS os buffers temporários
            dh_next = _tensorPool!.Rent(new[] { 1, hiddenSize });
            dc_next = _tensorPool.Rent(new[] { 1, hiddenSize });
            dy = _tensorPool.Rent(new[] { sequenceLength, outputSize });
            d_embedding = _tensorPool.Rent(new[] { 1, weightsEmbedding!.Shape[1] });
            current_dy = _tensorPool.Rent(new[] { 1, outputSize });
            dh = _tensorPool.Rent(new[] { 1, hiddenSize });
            dc = _tensorPool.Rent(new[] { 1, hiddenSize });
            dog = _tensorPool.Rent(new[] { 1, hiddenSize });
            dfg = _tensorPool.Rent(new[] { 1, hiddenSize });
            dig = _tensorPool.Rent(new[] { 1, hiddenSize });
            dcc = _tensorPool.Rent(new[] { 1, hiddenSize });
            temp_deriv = _tensorPool.Rent(new[] { 1, hiddenSize });
            temp_mult = _tensorPool.Rent(new[] { 1, hiddenSize });
            temp_grad_why = _tensorPool.Rent(new[] { hiddenSize, outputSize });
            temp_grad_w_in = _tensorPool.Rent(new[] { weightsEmbedding.Shape[1], hiddenSize });
            temp_grad_w_hid = _tensorPool.Rent(new[] { hiddenSize, hiddenSize });

            // ✅ FASE 2: Calcula gradiente de output (dy)
            _mathEngine.Subtract(predictions, targets, dy);

            // ✅ FASE 3: Loop backward (T-1 → 0)
            for (int t = sequenceLength - 1; t >= 0; t--)
            {
                // Extrai gradiente de output para este timestep
                _mathEngine.Slice(dy, t, current_dy);

                // ✅ Cache local de timestep (será liberado no finally interno)
                Dictionary<string, IMathTensor>? timestepCache = null;

                try
                {
                    // ✅ Carrega todos os tensores deste timestep DE UMA VEZ
                    timestepCache = _cacheManager.RetrieveMultipleTensors(t,
                        DiskOnlyCacheManager.TensorNames.Input,
                        DiskOnlyCacheManager.TensorNames.HiddenPrev,
                        DiskOnlyCacheManager.TensorNames.HiddenNext,
                        DiskOnlyCacheManager.TensorNames.CellPrev,
                        DiskOnlyCacheManager.TensorNames.CellNext,
                        DiskOnlyCacheManager.TensorNames.ForgetGate,
                        DiskOnlyCacheManager.TensorNames.InputGate,
                        DiskOnlyCacheManager.TensorNames.CellCandidate,
                        DiskOnlyCacheManager.TensorNames.OutputGate,
                        DiskOnlyCacheManager.TensorNames.TanhCellNext
                    );

                    // ✅ Gradiente de Output Final
                    var hiddenNext = timestepCache[DiskOnlyCacheManager.TensorNames.HiddenNext];
                    _mathEngine.MatrixMultiplyTransposeA(hiddenNext, current_dy, temp_grad_why);
                    _mathEngine.Add(grads["why"], temp_grad_why, grads["why"]);
                    _mathEngine.Add(grads["by"], current_dy, grads["by"]);

                    // ✅ Gradiente de Hidden State
                    _mathEngine.MatrixMultiplyTransposeB(current_dy, weightsHiddenOutputFinal!, dh);
                    _mathEngine.Add(dh, dh_next!, dh);

                    // ✅ Gradiente de Output Gate
                    var tanhCellNext = timestepCache[DiskOnlyCacheManager.TensorNames.TanhCellNext];
                    var outputGate = timestepCache[DiskOnlyCacheManager.TensorNames.OutputGate];

                    _mathEngine.Multiply(dh, tanhCellNext, dog);
                    _mathEngine.SigmoidDerivative(outputGate, temp_deriv);
                    _mathEngine.Multiply(dog, temp_deriv, dog);

                    // Atualiza pesos do Output Gate
                    var input = timestepCache[DiskOnlyCacheManager.TensorNames.Input];
                    var hiddenPrev = timestepCache[DiskOnlyCacheManager.TensorNames.HiddenPrev];

                    _mathEngine.MatrixMultiplyTransposeA(input, dog, temp_grad_w_in);
                    _mathEngine.Add(grads["wio"], temp_grad_w_in, grads["wio"]);
                    _mathEngine.MatrixMultiplyTransposeA(hiddenPrev, dog, temp_grad_w_hid);
                    _mathEngine.Add(grads["who"], temp_grad_w_hid, grads["who"]);
                    _mathEngine.Add(grads["bo"], dog, grads["bo"]);

                    // ✅ Gradiente de Cell State
                    _mathEngine.Multiply(dh, outputGate, temp_mult);
                    _mathEngine.TanhDerivative(tanhCellNext, temp_deriv);
                    _mathEngine.Multiply(temp_mult, temp_deriv, temp_mult);
                    _mathEngine.Add(dc_next!, temp_mult, dc);

                    // ✅ Gradiente de Forget Gate
                    var cellPrev = timestepCache[DiskOnlyCacheManager.TensorNames.CellPrev];
                    var forgetGate = timestepCache[DiskOnlyCacheManager.TensorNames.ForgetGate];

                    _mathEngine.Multiply(dc, cellPrev, dfg);
                    _mathEngine.SigmoidDerivative(forgetGate, temp_deriv);
                    _mathEngine.Multiply(dfg, temp_deriv, dfg);

                    _mathEngine.MatrixMultiplyTransposeA(input, dfg, temp_grad_w_in);
                    _mathEngine.Add(grads["wif"], temp_grad_w_in, grads["wif"]);
                    _mathEngine.MatrixMultiplyTransposeA(hiddenPrev, dfg, temp_grad_w_hid);
                    _mathEngine.Add(grads["whf"], temp_grad_w_hid, grads["whf"]);
                    _mathEngine.Add(grads["bf"], dfg, grads["bf"]);

                    // ✅ Gradiente de Input Gate
                    var cellCandidate = timestepCache[DiskOnlyCacheManager.TensorNames.CellCandidate];
                    var inputGate = timestepCache[DiskOnlyCacheManager.TensorNames.InputGate];

                    _mathEngine.Multiply(dc, cellCandidate, dig);
                    _mathEngine.SigmoidDerivative(inputGate, temp_deriv);
                    _mathEngine.Multiply(dig, temp_deriv, dig);

                    _mathEngine.MatrixMultiplyTransposeA(input, dig, temp_grad_w_in);
                    _mathEngine.Add(grads["wii"], temp_grad_w_in, grads["wii"]);
                    _mathEngine.MatrixMultiplyTransposeA(hiddenPrev, dig, temp_grad_w_hid);
                    _mathEngine.Add(grads["whi"], temp_grad_w_hid, grads["whi"]);
                    _mathEngine.Add(grads["bi"], dig, grads["bi"]);

                    // ✅ Gradiente de Cell Candidate
                    _mathEngine.Multiply(dc, inputGate, dcc);
                    _mathEngine.TanhDerivative(cellCandidate, temp_deriv);
                    _mathEngine.Multiply(dcc, temp_deriv, dcc);

                    _mathEngine.MatrixMultiplyTransposeA(input, dcc, temp_grad_w_in);
                    _mathEngine.Add(grads["wic"], temp_grad_w_in, grads["wic"]);
                    _mathEngine.MatrixMultiplyTransposeA(hiddenPrev, dcc, temp_grad_w_hid);
                    _mathEngine.Add(grads["whc"], temp_grad_w_hid, grads["whc"]);
                    _mathEngine.Add(grads["bc"], dcc, grads["bc"]);

                    // ✅ Gradiente de Embedding
                    _mathEngine.MatrixMultiplyTransposeB(dfg, weightsInputForget!, d_embedding);
                    _mathEngine.MatrixMultiplyTransposeB(dig, weightsInputInput!, temp_mult);
                    _mathEngine.Add(d_embedding, temp_mult, d_embedding);
                    _mathEngine.MatrixMultiplyTransposeB(dcc, weightsInputCell!, temp_mult);
                    _mathEngine.Add(d_embedding, temp_mult, d_embedding);
                    _mathEngine.MatrixMultiplyTransposeB(dog, weightsInputOutput!, temp_mult);
                    _mathEngine.Add(d_embedding, temp_mult, d_embedding);

                    _mathEngine.AccumulateGradient(grads["w_embed"], d_embedding, inputIndices[t]);

                    // ✅ Propaga gradientes para timestep anterior
                    _mathEngine.MatrixMultiplyTransposeB(dfg, weightsHiddenForget!, dh_next!);
                    _mathEngine.MatrixMultiplyTransposeB(dig, weightsHiddenInput!, temp_mult);
                    _mathEngine.Add(dh_next!, temp_mult, dh_next!);
                    _mathEngine.MatrixMultiplyTransposeB(dcc, weightsHiddenCell!, temp_mult);
                    _mathEngine.Add(dh_next!, temp_mult, dh_next!);
                    _mathEngine.MatrixMultiplyTransposeB(dog, weightsHiddenOutput!, temp_mult);
                    _mathEngine.Add(dh_next!, temp_mult, dh_next!);

                    _mathEngine.Multiply(dc, forgetGate, dc_next!);
                }
                finally
                {
                    // ✅ CRÍTICO: Libera tensores do timestep
                    if (timestepCache != null)
                    {
                        foreach (var tensor in timestepCache.Values)
                        {
                            tensor?.Dispose();
                        }

                        timestepCache.Clear();
                    }
                }
            }

            // ✅ FASE 4: Gradient Clipping
            double totalNorm = 0;
            foreach (var grad in grads.Values)
            {
                var gradData = grad.ToCpuTensor().GetData();
                totalNorm += gradData.Sum(x => x * x);
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

            // ✅ FASE 5: Retorna gradientes
            // ATENÇÃO: O chamador (TrainSequence) DEVE fazer Dispose destes gradientes!
            return grads;
        }
        catch (Exception ex)
        {
            // ✅ EM CASO DE EXCEÇÃO: Libera gradientes para evitar vazamento
            Console.WriteLine($"[ERRO] BackwardPass falhou: {ex.Message}");

            // Libera gradientes criados
            if (grads != null)
            {
                foreach (var grad in grads.Values)
                {
                    try
                    {
                        grad?.Dispose();
                    }
                    catch
                    {
                    }
                }

                grads.Clear();
            }

            throw;
        }
        finally
        {
            // ✅ GARANTIA: Retorna TODOS os buffers temporários ao pool
            if (dh_next != null) _tensorPool.Return(dh_next);
            if (dc_next != null) _tensorPool.Return(dc_next);
            if (dy != null) _tensorPool.Return(dy);
            if (d_embedding != null) _tensorPool.Return(d_embedding);
            if (current_dy != null) _tensorPool.Return(current_dy);
            if (dh != null) _tensorPool.Return(dh);
            if (dc != null) _tensorPool.Return(dc);
            if (dog != null) _tensorPool.Return(dog);
            if (dfg != null) _tensorPool.Return(dfg);
            if (dig != null) _tensorPool.Return(dig);
            if (dcc != null) _tensorPool.Return(dcc);
            if (temp_deriv != null) _tensorPool.Return(temp_deriv);
            if (temp_mult != null) _tensorPool.Return(temp_mult);
            if (temp_grad_why != null) _tensorPool.Return(temp_grad_why);
            if (temp_grad_w_in != null) _tensorPool.Return(temp_grad_w_in);
            if (temp_grad_w_hid != null) _tensorPool.Return(temp_grad_w_hid);

            // ⚠️ ATENÇÃO: NÃO liberamos 'grads' aqui porque será usado em UpdateWeights!
            // O chamador (TrainSequence) é responsável por liberar após uso.
        }
    }

    private void UpdateWeightsWithAdamGpu(Dictionary<string, IMathTensor> gradients, double learningRate)
    {
        var parameters = new Dictionary<string, IMathTensor>
        {
            { "w_embed", weightsEmbedding! }, { "wif", weightsInputForget! }, { "whf", weightsHiddenForget! },
            { "wii", weightsInputInput! }, { "whi", weightsHiddenInput! }, { "wic", weightsInputCell! },
            { "whc", weightsHiddenCell! }, { "wio", weightsInputOutput! }, { "who", weightsHiddenOutput! },
            { "why", weightsHiddenOutputFinal! }, { "bf", biasForget! }, { "bi", biasInput! },
            { "bc", biasCell! }, { "bo", biasOutput! }, { "by", biasOutputFinal! }
        };

        int layerId = 0;
        double[]? paramData = null;
        double[]? gradData = null;

        try
        {
            foreach (var key in parameters.Keys.Where(gradients.ContainsKey))
            {
                var paramTensor = parameters[key];
                var gradTensor = gradients[key];

                paramData = paramTensor.ToCpuTensor().GetData();
                gradData = gradTensor.ToCpuTensor().GetData();

                if (paramData.Length != gradData.Length)
                {
                    throw new InvalidOperationException(
                        $"Incompatibilidade de tamanho em '{key}': Param={paramData.Length}, Grad={gradData.Length}");
                }

                _adamOptimizer.UpdateParameters(layerId, paramData, gradData);
                paramTensor.UpdateFromCpu(paramData);

                paramData = null;
                gradData = null;
                layerId++;
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[ERRO] UpdateWeights falhou na layer {layerId}: {ex.Message}");
            throw;
        }
        finally
        {
            paramData = null;
            gradData = null;
            GC.Collect(0, GCCollectionMode.Optimized, false);
        }
    }

    // ========================================
// INITIALIZE GRADIENTS GPU - AUDITORIA
// ========================================

    private Dictionary<string, IMathTensor> InitializeGradientsGpu()
    {
        int embeddingSize = weightsEmbedding!.Shape[1];

        // ✅ Cria dicionário com 15 tensores
        var gradients = new Dictionary<string, IMathTensor>
        {
            // ✅ Embedding (maior tensor: vocabSize × embeddingSize)
            { "w_embed", _mathEngine.CreateTensor(new[] { inputSize, embeddingSize }) },

            // ✅ LSTM Weights - Input Gates (8 tensores)
            { "wif", _mathEngine.CreateTensor(new[] { embeddingSize, hiddenSize }) },
            { "whf", _mathEngine.CreateTensor(new[] { hiddenSize, hiddenSize }) },
            { "wii", _mathEngine.CreateTensor(new[] { embeddingSize, hiddenSize }) },
            { "whi", _mathEngine.CreateTensor(new[] { hiddenSize, hiddenSize }) },
            { "wic", _mathEngine.CreateTensor(new[] { embeddingSize, hiddenSize }) },
            { "whc", _mathEngine.CreateTensor(new[] { hiddenSize, hiddenSize }) },
            { "wio", _mathEngine.CreateTensor(new[] { embeddingSize, hiddenSize }) },
            { "who", _mathEngine.CreateTensor(new[] { hiddenSize, hiddenSize }) },

            // ✅ Output Layer (segundo maior tensor: hiddenSize × outputSize)
            { "why", _mathEngine.CreateTensor(new[] { hiddenSize, outputSize }) },

            // ✅ Biases (5 tensores pequenos)
            { "bf", _mathEngine.CreateTensor(new[] { 1, hiddenSize }) },
            { "bi", _mathEngine.CreateTensor(new[] { 1, hiddenSize }) },
            { "bc", _mathEngine.CreateTensor(new[] { 1, hiddenSize }) },
            { "bo", _mathEngine.CreateTensor(new[] { 1, hiddenSize }) },
            { "by", _mathEngine.CreateTensor(new[] { 1, outputSize }) }
        };

        // ⚠️ AVISO CRÍTICO:
        // Estes 15 tensores DEVEM ser liberados após uso!
        // O chamador (BackwardPass) retorna este dicionário,
        // e quem recebe (TrainSequence) DEVE fazer Dispose de TODOS.

        return gradients;
    }
// ========================================
// CÁLCULO DE MEMÓRIA DOS GRADIENTES
// ========================================

    private long CalculateGradientsMemoryMB()
    {
        int embeddingSize = weightsEmbedding!.Shape[1];
        long totalParams = 0;

        // Embedding
        totalParams += inputSize * embeddingSize;

        // LSTM Weights (4 gates × 2 tipos)
        totalParams += 4 * embeddingSize * hiddenSize; // Input weights
        totalParams += 4 * hiddenSize * hiddenSize; // Hidden weights

        // Output Layer
        totalParams += hiddenSize * outputSize;

        // Biases
        totalParams += 4 * hiddenSize; // LSTM biases
        totalParams += outputSize; // Output bias

        // Cada parâmetro: 8 bytes (double)
        long totalBytes = totalParams * sizeof(double);
        long totalMB = totalBytes / (1024 * 1024);

        return totalMB;
    }
// ========================================
// MÉTODO AUXILIAR: Libera Gradientes
// ========================================

    /// <summary>
    /// Libera TODOS os tensores de um dicionário de gradientes.
    /// Use este método para garantir que nenhum tensor fique órfão.
    /// </summary>
    private void DisposeGradients(Dictionary<string, IMathTensor>? gradients)
    {
        if (gradients == null) return;

        foreach (var kvp in gradients)
        {
            try
            {
                kvp.Value?.Dispose();
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[AVISO] Erro ao liberar gradiente '{kvp.Key}': {ex.Message}");
            }
        }

        gradients.Clear();
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

    public double CalculateSequenceLoss(int[] inputIndices, Tensor[] targets)
    {
        _cacheManager.Reset();
        using var targetsGpu = CreateSequenceTensor(targets);
        var (predictions, loss) = ForwardPassGpuOptimized(inputIndices, targetsGpu, inputIndices.Length);
        predictions.Dispose();
        return loss;
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
            var why = mathEngine.CreateTensor(modelData.WeightsHiddenOutputFinal.data,
                modelData.WeightsHiddenOutputFinal.shape);
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

    public void Dispose()
    {
        Dispose(true);
        GC.SuppressFinalize(this);
    }
}