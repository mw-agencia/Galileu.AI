using System.Text.Json;
using Galileu.Node.Core;
using Galileu.Node.Brain.Gpu;
using OpenCL.Net;
using System.Linq;
using System;
using System.Collections.Generic;

namespace Galileu.Node.Brain;

public class NeuralNetworkLSTM
{
    // ... (campos, propriedades e construtores permanecem os mesmos) ...
    private Tensor weightsInputForget;
    private Tensor weightsHiddenForget;
    private Tensor weightsInputInput;
    private Tensor weightsHiddenInput;
    private Tensor weightsInputCell;
    private Tensor weightsHiddenCell;
    private Tensor weightsInputOutput;
    private Tensor weightsHiddenOutput;
    private Tensor biasForget;
    private Tensor biasInput;
    private Tensor biasCell;
    private Tensor biasOutput;
    private Tensor weightsHiddenOutputFinal;
    private Tensor biasOutputFinal;
    private readonly int inputSize;
    private readonly int hiddenSize;
    private readonly int outputSize;
    private double[] hiddenState;
    private double[] cellState;

    private readonly OpenCLService? _openCLService;
    private readonly bool _useGpu;
    private Dictionary<string, GpuTensor> _gpuWeights = new();
    private GpuTensor? _hiddenStateGpu;
    private GpuTensor? _cellStateGpu;

    public int InputSize => inputSize;
    public int HiddenSize => hiddenSize;
    public int OutputSize => outputSize;

    public Tensor WeightsInputForget => weightsInputForget;
    public Tensor WeightsHiddenForget => weightsHiddenForget;
    public Tensor WeightsInputInput => weightsInputInput;
    public Tensor WeightsHiddenInput => weightsHiddenInput;
    public Tensor WeightsInputCell => weightsInputCell;
    public Tensor WeightsHiddenCell => weightsHiddenCell;
    public Tensor WeightsInputOutput => weightsInputOutput;
    public Tensor WeightsHiddenOutput => weightsHiddenOutput;
    public Tensor BiasForget => biasForget;
    public Tensor BiasInput => biasInput;
    public Tensor BiasCell => biasCell;
    public Tensor BiasOutput => biasOutput;
    public Tensor WeightsHiddenOutputFinal => weightsHiddenOutputFinal;
    public Tensor BiasOutputFinal => biasOutputFinal;

    public NeuralNetworkLSTM(int inputSize, int hiddenSize, int outputSize, OpenCLService openCLService)
    {
        _openCLService = openCLService;
        _useGpu = _openCLService?.IsGpuAvailable ?? false;

        this.inputSize = inputSize;
        this.hiddenSize = hiddenSize;
        this.outputSize = outputSize;
        hiddenState = new double[hiddenSize];
        cellState = new double[hiddenSize];

        Random rand = new Random();
        weightsInputForget = InitializeTensor(inputSize, hiddenSize, rand);
        weightsHiddenForget = InitializeTensor(hiddenSize, hiddenSize, rand);
        weightsInputInput = InitializeTensor(inputSize, hiddenSize, rand);
        weightsHiddenInput = InitializeTensor(hiddenSize, hiddenSize, rand);
        weightsInputCell = InitializeTensor(inputSize, hiddenSize, rand);
        weightsHiddenCell = InitializeTensor(hiddenSize, hiddenSize, rand);
        weightsInputOutput = InitializeTensor(inputSize, hiddenSize, rand);
        weightsHiddenOutput = InitializeTensor(hiddenSize, hiddenSize, rand);
        biasForget = InitializeTensor(hiddenSize, rand);
        biasInput = InitializeTensor(hiddenSize, rand);
        biasCell = InitializeTensor(hiddenSize, rand);
        biasOutput = InitializeTensor(hiddenSize, rand);
        weightsHiddenOutputFinal = InitializeTensor(hiddenSize, outputSize, rand);
        biasOutputFinal = InitializeTensor(outputSize, rand);

        if (_useGpu)
        {
            InitializeGpuTensors();
        }
    }
    
    protected NeuralNetworkLSTM(int inputSize, int hiddenSize, int outputSize,
                           Tensor weightsInputForget, Tensor weightsHiddenForget,
                           Tensor weightsInputInput, Tensor weightsHiddenInput,
                           Tensor weightsInputCell, Tensor weightsHiddenCell,
                           Tensor weightsInputOutput, Tensor weightsHiddenOutput,
                           Tensor biasForget, Tensor biasInput, Tensor biasCell, Tensor biasOutput,
                           Tensor weightsHiddenOutputFinal, Tensor biasOutputFinal, OpenCLService openCLService)
    {
        _openCLService = openCLService;
        _useGpu = _openCLService?.IsGpuAvailable ?? false;

        this.inputSize = inputSize;
        this.hiddenSize = hiddenSize;
        this.outputSize = outputSize;
        this.hiddenState = new double[hiddenSize];
        this.cellState = new double[hiddenSize];
        this.weightsInputForget = weightsInputForget ?? throw new ArgumentNullException(nameof(weightsInputForget));
        this.weightsHiddenForget = weightsHiddenForget ?? throw new ArgumentNullException(nameof(weightsHiddenForget));
        this.weightsInputInput = weightsInputInput ?? throw new ArgumentNullException(nameof(weightsInputInput));
        this.weightsHiddenInput = weightsHiddenInput ?? throw new ArgumentNullException(nameof(weightsHiddenInput));
        this.weightsInputCell = weightsInputCell ?? throw new ArgumentNullException(nameof(weightsInputCell));
        this.weightsHiddenCell = weightsHiddenCell ?? throw new ArgumentNullException(nameof(weightsHiddenCell));
        this.weightsInputOutput = weightsInputOutput ?? throw new ArgumentNullException(nameof(weightsInputOutput));
        this.weightsHiddenOutput = weightsHiddenOutput ?? throw new ArgumentNullException(nameof(weightsHiddenOutput));
        this.biasForget = biasForget ?? throw new ArgumentNullException(nameof(biasForget));
        this.biasInput = biasInput ?? throw new ArgumentNullException(nameof(biasInput));
        this.biasCell = biasCell ?? throw new ArgumentNullException(nameof(biasCell));
        this.biasOutput = biasOutput ?? throw new ArgumentNullException(nameof(biasOutput));
        this.weightsHiddenOutputFinal = weightsHiddenOutputFinal ?? throw new ArgumentNullException(nameof(weightsHiddenOutputFinal));
        this.biasOutputFinal = biasOutputFinal ?? throw new ArgumentNullException(nameof(biasOutputFinal));
        
        if (_useGpu)
        {
            InitializeGpuTensors();
        }
    }
    // ... (Forward, ForwardGpu, ForwardCpu e métodos helper de GPU permanecem os mesmos) ...
    
    private void InitializeGpuTensors()
    {
        if (_openCLService == null) return;
        
        _gpuWeights["weightsInputForget"] = CreateGpuTensor(weightsInputForget);
        _gpuWeights["weightsHiddenForget"] = CreateGpuTensor(weightsHiddenForget);
        _gpuWeights["weightsInputInput"] = CreateGpuTensor(weightsInputInput);
        _gpuWeights["weightsHiddenInput"] = CreateGpuTensor(weightsHiddenInput);
        _gpuWeights["weightsInputCell"] = CreateGpuTensor(weightsInputCell);
        _gpuWeights["weightsHiddenCell"] = CreateGpuTensor(weightsHiddenCell);
        _gpuWeights["weightsInputOutput"] = CreateGpuTensor(weightsInputOutput);
        _gpuWeights["weightsHiddenOutput"] = CreateGpuTensor(weightsHiddenOutput);
        _gpuWeights["biasForget"] = CreateGpuTensor(biasForget);
        _gpuWeights["biasInput"] = CreateGpuTensor(biasInput);
        _gpuWeights["biasCell"] = CreateGpuTensor(biasCell);
        _gpuWeights["biasOutput"] = CreateGpuTensor(biasOutput);
        _gpuWeights["weightsHiddenOutputFinal"] = CreateGpuTensor(weightsHiddenOutputFinal);
        _gpuWeights["biasOutputFinal"] = CreateGpuTensor(biasOutputFinal);
        
        _hiddenStateGpu = new GpuTensor(_openCLService, 1, hiddenSize);
        _cellStateGpu = new GpuTensor(_openCLService, 1, hiddenSize);
        ResetHiddenState();
    }
    
    private GpuTensor CreateGpuTensor(Tensor cpuTensor)
    {
        if (_openCLService == null) throw new InvalidOperationException("OpenCLService not available.");
        
        var shape = cpuTensor.GetShape();
        int rows = shape.Length > 1 ? shape[0] : 1;
        int cols = shape.Length > 1 ? shape[1] : shape[0];

        var gpuTensor = new GpuTensor(_openCLService, rows, cols);
        gpuTensor.Write(cpuTensor.GetData());
        return gpuTensor;
    }
    
    private Tensor InitializeTensor(int rows, int cols, Random rand)
    {
        double[] data = new double[rows * cols];
        for (int i = 0; i < data.Length; i++)
            data[i] = rand.NextDouble() - 0.5;
        return new Tensor(data, new int[] { rows, cols });
    }

    private Tensor InitializeTensor(int size, Random rand)
    {
        double[] data = new double[size];
        for (int i = 0; i < data.Length; i++)
            data[i] = rand.NextDouble() - 0.5;
        return new Tensor(data, new int[] { size });
    }

    public Tensor Forward(Tensor input)
    {
        if (_useGpu)
        {
            return ForwardGpu(input);
        }
        else
        {
            return ForwardCpu(input);
        }
    }

    private Tensor ForwardGpu(Tensor input)
    {
        if (_openCLService == null || !_openCLService.CommandQueue.HasValue || _hiddenStateGpu == null || _cellStateGpu == null) throw new InvalidOperationException("GPU resources not initialized.");

        using var inputGpu = CreateGpuTensor(input);
        
        using var f_t = new GpuTensor(_openCLService, 1, hiddenSize);
        using var i_t = new GpuTensor(_openCLService, 1, hiddenSize);
        using var C_tilde = new GpuTensor(_openCLService, 1, hiddenSize);
        using var o_t = new GpuTensor(_openCLService, 1, hiddenSize);
        
        ComputeGateGpu("sigmoid_forward", inputGpu, _hiddenStateGpu, "weightsInputForget", "weightsHiddenForget", "biasForget", f_t);
        ComputeGateGpu("sigmoid_forward", inputGpu, _hiddenStateGpu, "weightsInputInput", "weightsHiddenInput", "biasInput", i_t);
        ComputeGateGpu("tanh_forward", inputGpu, _hiddenStateGpu, "weightsInputCell", "weightsHiddenCell", "biasCell", C_tilde);
        ComputeGateGpu("sigmoid_forward", inputGpu, _hiddenStateGpu, "weightsInputOutput", "weightsHiddenOutput", "biasOutput", o_t);
        
        using var newCellStateGpu = new GpuTensor(_openCLService, 1, hiddenSize);
        using var newHiddenStateGpu = new GpuTensor(_openCLService, 1, hiddenSize);

        using (var temp1 = new GpuTensor(_openCLService, 1, hiddenSize))
        using (var temp2 = new GpuTensor(_openCLService, 1, hiddenSize))
        {
            ExecuteElementwiseKernel("elementwise_multiply", f_t.Buffer, _cellStateGpu.Buffer, temp1.Buffer, hiddenSize);
            ExecuteElementwiseKernel("elementwise_multiply", i_t.Buffer, C_tilde.Buffer, temp2.Buffer, hiddenSize);
            ExecuteElementwiseKernel("elementwise_add_forward", temp1.Buffer, temp2.Buffer, newCellStateGpu.Buffer, hiddenSize);
        }
        
        using (var tempTanh = new GpuTensor(_openCLService, 1, hiddenSize))
        {
            ExecuteElementwiseKernel("tanh_forward", newCellStateGpu.Buffer, null, tempTanh.Buffer, hiddenSize);
            ExecuteElementwiseKernel("elementwise_multiply", o_t.Buffer, tempTanh.Buffer, newHiddenStateGpu.Buffer, hiddenSize);
        }
        
        // CORREÇÃO: O estado é salvo na CPU e GPU para o backpropagation.
        hiddenState = newHiddenStateGpu.Read();
        cellState = newCellStateGpu.Read();

        _hiddenStateGpu.Dispose();
        _cellStateGpu.Dispose();
        _hiddenStateGpu = newHiddenStateGpu;
        _cellStateGpu = newCellStateGpu;
        
        using var outputLogits = new GpuTensor(_openCLService, 1, outputSize);
        ExecuteMatMulKernel(_hiddenStateGpu.Buffer, _gpuWeights["weightsHiddenOutputFinal"].Buffer, outputLogits.Buffer, 1, hiddenSize, outputSize);
        ExecuteBroadcastAddKernel(outputLogits.Buffer, _gpuWeights["biasOutputFinal"].Buffer, outputLogits.Buffer, 1, outputSize);
        
        using var expOutput = new GpuTensor(_openCLService, 1, outputSize);
        ExecuteElementwiseKernel("exp_forward", outputLogits.Buffer, null, expOutput.Buffer, outputSize);
        
        if (_openCLService.CommandQueue.HasValue)
        {
            Cl.Finish(_openCLService.CommandQueue.Value);
        }
        
        double[] outputData = expOutput.Read();
        double sumExp = outputData.Sum();
        if (sumExp == 0) sumExp = 1e-9;
        for (int o = 0; o < outputSize; o++)
        {
            outputData[o] /= sumExp;
        }

        return new Tensor(outputData, new int[] { outputSize });
    }
    
    private Tensor ForwardCpu(Tensor input)
    {
        if (input == null || input.GetShape().Length != 1 || input.GetShape()[0] != inputSize)
        {
            throw new ArgumentException("Entrada deve ser unidimensional com tamanho inputSize.");
        }

        double[] newHiddenState = new double[hiddenSize];
        double[] newCellState = new double[hiddenSize];

        for (int h = 0; h < hiddenSize; h++)
        {
            double f_t = Sigmoid(ComputeGate(input, weightsInputForget, hiddenState, weightsHiddenForget, biasForget, h));
            double i_t = Sigmoid(ComputeGate(input, weightsInputInput, hiddenState, weightsHiddenInput, biasInput, h));
            double C_tilde = Math.Tanh(ComputeGate(input, weightsInputCell, hiddenState, weightsHiddenCell, biasCell, h));
            double o_t = Sigmoid(ComputeGate(input, weightsInputOutput, hiddenState, weightsHiddenOutput, biasOutput, h));
            
            newCellState[h] = f_t * cellState[h] + i_t * C_tilde;
            newHiddenState[h] = o_t * Math.Tanh(newCellState[h]);
        }

        hiddenState = newHiddenState;
        cellState = newCellState;
        
        double[] outputData = new double[outputSize];
        double sumExp = 0;
        for (int o = 0; o < outputSize; o++)
        {
            double sum = 0;
            for (int h = 0; h < hiddenSize; h++)
            {
                sum += hiddenState[h] * weightsHiddenOutputFinal.Infer(new int[] { h, o });
            }
            sum += biasOutputFinal.Infer(new int[] { o });
            outputData[o] = Math.Exp(sum);
            sumExp += outputData[o];
        }

        if (sumExp == 0) sumExp = 1e-9;
        for (int o = 0; o < outputSize; o++)
        {
            outputData[o] /= sumExp;
        }

        return new Tensor(outputData, new int[] { outputSize });
    }

    private double Sigmoid(double x)
    {
        return 1.0 / (1.0 + Math.Exp(-x));
    }

    private double ComputeGate(Tensor input, Tensor weightsInput, double[] prevHidden, Tensor weightsHidden, Tensor bias, int h)
    {
        double sum = 0;
        for (int i = 0; i < inputSize; i++)
        {
            sum += input.Infer(new int[] { i }) * weightsInput.Infer(new int[] { i, h });
        }
        for (int prevH = 0; prevH < hiddenSize; prevH++)
        {
            sum += prevHidden[prevH] * weightsHidden.Infer(new int[] { prevH, h });
        }
        sum += bias.Infer(new int[] { h });
        return sum;
    }

    private void ComputeGateGpu(string activationKernel, GpuTensor input, GpuTensor hidden, string wInKey, string wHidKey, string biasKey, GpuTensor result)
    {
        if (_openCLService == null) return;
        
        using var temp1 = new GpuTensor(_openCLService, 1, hiddenSize);
        using var temp2 = new GpuTensor(_openCLService, 1, hiddenSize);
        using var sum = new GpuTensor(_openCLService, 1, hiddenSize);

        ExecuteMatMulKernel(input.Buffer, _gpuWeights[wInKey].Buffer, temp1.Buffer, 1, inputSize, hiddenSize);
        ExecuteMatMulKernel(hidden.Buffer, _gpuWeights[wHidKey].Buffer, temp2.Buffer, 1, hiddenSize, hiddenSize);
        ExecuteElementwiseKernel("elementwise_add_forward", temp1.Buffer, temp2.Buffer, sum.Buffer, hiddenSize);
        ExecuteBroadcastAddKernel(sum.Buffer, _gpuWeights[biasKey].Buffer, sum.Buffer, 1, hiddenSize);
        ExecuteElementwiseKernel(activationKernel, sum.Buffer, null, result.Buffer, hiddenSize);
    }

    private void ExecuteMatMulKernel(IMem A, IMem B, IMem C, int M, int K, int N)
    {
        if (_openCLService == null || !_openCLService.CommandQueue.HasValue) return;

        var kernel = _openCLService.Kernels["matmul_forward"];
        
        Cl.SetKernelArg(kernel, 0u, (IntPtr)IntPtr.Size, A);
        Cl.SetKernelArg(kernel, 1u, (IntPtr)IntPtr.Size, B);
        Cl.SetKernelArg(kernel, 2u, (IntPtr)IntPtr.Size, C);
        Cl.SetKernelArg(kernel, 3u, (IntPtr)sizeof(int), M);
        Cl.SetKernelArg(kernel, 4u, (IntPtr)sizeof(int), K);
        Cl.SetKernelArg(kernel, 5u, (IntPtr)sizeof(int), N);
        
        Cl.EnqueueNDRangeKernel(_openCLService.CommandQueue.Value, kernel, 2, null, new IntPtr[] { (IntPtr)N, (IntPtr)M }, null, 0, null, out _);
    }
    
    private void ExecuteElementwiseKernel(string kernelName, IMem A, IMem? B, IMem C, int size)
    {
        if (_openCLService == null || !_openCLService.CommandQueue.HasValue) return;

        var kernel = _openCLService.Kernels[kernelName];
        uint argIndex = 0;
        
        Cl.SetKernelArg(kernel, argIndex++, (IntPtr)IntPtr.Size, A);
        if (B != null)
        {
            Cl.SetKernelArg(kernel, argIndex++, (IntPtr)IntPtr.Size, B);
        }
        Cl.SetKernelArg(kernel, argIndex++, (IntPtr)IntPtr.Size, C);
        Cl.SetKernelArg(kernel, argIndex, (IntPtr)sizeof(int), size);

        Cl.EnqueueNDRangeKernel(_openCLService.CommandQueue.Value, kernel, 1, null, new IntPtr[] { (IntPtr)size }, null, 0, null, out _);
    }
    
    private void ExecuteBroadcastAddKernel(IMem A, IMem B, IMem C, int M, int N)
    {
        if (_openCLService == null || !_openCLService.CommandQueue.HasValue) return;

        var kernel = _openCLService.Kernels["elementwise_add_broadcast_forward"];
        
        Cl.SetKernelArg(kernel, 0u, (IntPtr)IntPtr.Size, A);
        Cl.SetKernelArg(kernel, 1u, (IntPtr)IntPtr.Size, B);
        Cl.SetKernelArg(kernel, 2u, (IntPtr)IntPtr.Size, C);
        Cl.SetKernelArg(kernel, 3u, (IntPtr)sizeof(int), M);
        Cl.SetKernelArg(kernel, 4u, (IntPtr)sizeof(int), N);

        Cl.EnqueueNDRangeKernel(_openCLService.CommandQueue.Value, kernel, 2, null, new IntPtr[] { (IntPtr)N, (IntPtr)M }, null, 0, null, out _);
    }
    
    public void ResetHiddenState()
    {
        hiddenState = new double[hiddenSize];
        cellState = new double[hiddenSize];
        if (_useGpu)
        {
            _hiddenStateGpu?.Write(hiddenState);
            _cellStateGpu?.Write(cellState);
        }
    }
    
    // ####################################################################
    // ## MÉTODO TrainEpoch MODIFICADO PARA USAR GPU                     ##
    // ####################################################################

    public double TrainEpoch(Tensor[] inputs, Tensor[] targets, double learningRate)
    {
        double epochLoss = 0;

        for (int i = 0; i < inputs.Length; i++)
        {
            // Reset do estado para cada amostra de treino
            ResetHiddenState(); 
            
            // CORREÇÃO: O passo Forward agora usa a GPU se disponível
            Tensor output = Forward(inputs[i]); 

            // Cálculo da perda (na CPU)
            for (int o = 0; o < outputSize; o++)
            {
                if (targets[i].Infer(new int[] { o }) == 1.0)
                {
                    epochLoss += -Math.Log(output.Infer(new int[] { o }) + 1e-9);
                    break;
                }
            }
            
            // Backpropagation (na CPU, usando os estados 'hiddenState' e 'cellState'
            // que foram atualizados e lidos de volta da GPU no método ForwardGpu).
            double[] gradOutput = new double[outputSize];
            for (int o = 0; o < outputSize; o++)
            {
                gradOutput[o] = output.Infer(new int[] { o }) - targets[i].Infer(new int[] { o });
            }

            // ... (resto da lógica de backpropagation permanece na CPU)
            double[] newWeightsOutputData = new double[hiddenSize * outputSize];
            double[] newBiasOutputData = new double[outputSize];
            for (int o = 0; o < outputSize; o++)
            {
                for (int h = 0; h < hiddenSize; h++)
                {
                    int idx = h * outputSize + o;
                    newWeightsOutputData[idx] = weightsHiddenOutputFinal.Infer(new int[] { h, o }) -
                                               learningRate * gradOutput[o] * hiddenState[h];
                }
                newBiasOutputData[o] = biasOutputFinal.Infer(new int[] { o }) - learningRate * gradOutput[o];
            }
            weightsHiddenOutputFinal = new Tensor(newWeightsOutputData, new int[] { hiddenSize, outputSize });
            biasOutputFinal = new Tensor(newBiasOutputData, new int[] { outputSize });

            double[] gradHidden = new double[hiddenSize];
            for (int h = 0; h < hiddenSize; h++)
            {
                double sum = 0;
                for (int o = 0; o < outputSize; o++)
                {
                    sum += gradOutput[o] * weightsHiddenOutputFinal.Infer(new int[] { h, o });
                }
                gradHidden[h] = sum * (1 - Math.Pow(Math.Tanh(cellState[h]), 2)); 
            }
            
            UpdateGateWeights(ref weightsInputForget, ref weightsHiddenForget, ref biasForget, inputs[i], gradHidden, learningRate);
            UpdateGateWeights(ref weightsInputInput, ref weightsHiddenInput, ref biasInput, inputs[i], gradHidden, learningRate);
            UpdateGateWeights(ref weightsInputCell, ref weightsHiddenCell, ref biasCell, inputs[i], gradHidden, learningRate);
            UpdateGateWeights(ref weightsInputOutput, ref weightsHiddenOutput, ref biasOutput, inputs[i], gradHidden, learningRate);
        
            // CORREÇÃO: Após cada atualização, os novos pesos precisam ser enviados para a GPU.
            if (_useGpu)
            {
                UpdateGpuWeights();
            }
        }
        return epochLoss / inputs.Length;
    }
    
    private void UpdateGpuWeights()
    {
        _gpuWeights["weightsInputForget"]?.Write(weightsInputForget.GetData());
        _gpuWeights["weightsHiddenForget"]?.Write(weightsHiddenForget.GetData());
        _gpuWeights["weightsInputInput"]?.Write(weightsInputInput.GetData());
        _gpuWeights["weightsHiddenInput"]?.Write(weightsHiddenInput.GetData());
        _gpuWeights["weightsInputCell"]?.Write(weightsInputCell.GetData());
        _gpuWeights["weightsHiddenCell"]?.Write(weightsHiddenCell.GetData());
        _gpuWeights["weightsInputOutput"]?.Write(weightsInputOutput.GetData());
        _gpuWeights["weightsHiddenOutput"]?.Write(weightsHiddenOutput.GetData());
        _gpuWeights["biasForget"]?.Write(biasForget.GetData());
        _gpuWeights["biasInput"]?.Write(biasInput.GetData());
        _gpuWeights["biasCell"]?.Write(biasCell.GetData());
        _gpuWeights["biasOutput"]?.Write(biasOutput.GetData());
        _gpuWeights["weightsHiddenOutputFinal"]?.Write(weightsHiddenOutputFinal.GetData());
        _gpuWeights["biasOutputFinal"]?.Write(biasOutputFinal.GetData());
    }

    private void UpdateGateWeights(ref Tensor weightsInput, ref Tensor weightsHidden, ref Tensor bias, Tensor input, double[] gradHidden, double learningRate)
    {
        var wInputData = weightsInput.GetData().ToArray();
        var wHiddenData = weightsHidden.GetData().ToArray();
        var biasData = bias.GetData().ToArray();

        for (int h = 0; h < hiddenSize; h++)
        {
            for (int j = 0; j < inputSize; j++)
            {
                wInputData[j * hiddenSize + h] -= learningRate * gradHidden[h] * input.Infer(new int[] { j });
            }
            for (int prevH = 0; prevH < hiddenSize; prevH++)
            {
                wHiddenData[prevH * hiddenSize + h] -= learningRate * gradHidden[h] * hiddenState[prevH];
            }
            biasData[h] -= learningRate * gradHidden[h];
        }

        weightsInput = new Tensor(wInputData, weightsInput.GetShape());
        weightsHidden = new Tensor(wHiddenData, weightsHidden.GetShape());
        bias = new Tensor(biasData, bias.GetShape());
    }
    
    // ... (SaveModel e LoadModel permanecem os mesmos) ...
    public void SaveModel(string filePath)
    {
        try
        {
            var modelData = new NeuralNetworkModelDataLSTM
            {
                InputSize = inputSize,
                HiddenSize = hiddenSize,
                OutputSize = outputSize,
                WeightsInputForget = new TensorData { data = weightsInputForget.GetData(), shape = weightsInputForget.GetShape() },
                WeightsHiddenForget = new TensorData { data = weightsHiddenForget.GetData(), shape = weightsHiddenForget.GetShape() },
                WeightsInputInput = new TensorData { data = weightsInputInput.GetData(), shape = weightsInputInput.GetShape() },
                WeightsHiddenInput = new TensorData { data = weightsHiddenInput.GetData(), shape = weightsHiddenInput.GetShape() },
                WeightsInputCell = new TensorData { data = weightsInputCell.GetData(), shape = weightsInputCell.GetShape() },
                WeightsHiddenCell = new TensorData { data = weightsHiddenCell.GetData(), shape = weightsHiddenCell.GetShape() },
                WeightsInputOutput = new TensorData { data = weightsInputOutput.GetData(), shape = weightsInputOutput.GetShape() },
                WeightsHiddenOutput = new TensorData { data = weightsHiddenOutput.GetData(), shape = weightsHiddenOutput.GetShape() },
                BiasForget = new TensorData { data = biasForget.GetData(), shape = biasForget.GetShape() },
                BiasInput = new TensorData { data = biasInput.GetData(), shape = biasInput.GetShape() },
                BiasCell = new TensorData { data = biasCell.GetData(), shape = biasCell.GetShape() },
                BiasOutput = new TensorData { data = biasOutput.GetData(), shape = biasOutput.GetShape() },
                WeightsHiddenOutputFinal = new TensorData { data = weightsHiddenOutputFinal.GetData(), shape = weightsHiddenOutputFinal.GetShape() },
                BiasOutputFinal = new TensorData { data = biasOutputFinal.GetData(), shape = biasOutputFinal.GetShape() }
            };

            var options = new JsonSerializerOptions { WriteIndented = true };
            string jsonString = JsonSerializer.Serialize(modelData, options);

            File.WriteAllText(filePath, jsonString);
            Console.WriteLine($"Modelo LSTM salvo em JSON em: {filePath}");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Erro ao salvar o modelo LSTM: {ex.Message}");
        }
    }

    public static NeuralNetworkLSTM? LoadModel(string filePath, OpenCLService openCLService)
    {
        try
        {
            if (!File.Exists(filePath))
            {
                Console.WriteLine($"Arquivo do modelo LSTM não encontrado em: {filePath}");
                return null;
            }

            string jsonString = File.ReadAllText(filePath);
            var modelData = JsonSerializer.Deserialize<NeuralNetworkModelDataLSTM>(jsonString);

            if (modelData == null)
            {
                throw new Exception("Falha ao desserializar dados do modelo LSTM.");
            }

            Tensor loadedWeightsInputForget = new Tensor(modelData.WeightsInputForget.data, modelData.WeightsInputForget.shape);
            Tensor loadedWeightsHiddenForget = new Tensor(modelData.WeightsHiddenForget.data, modelData.WeightsHiddenForget.shape);
            Tensor loadedWeightsInputInput = new Tensor(modelData.WeightsInputInput.data, modelData.WeightsInputInput.shape);
            Tensor loadedWeightsHiddenInput = new Tensor(modelData.WeightsHiddenInput.data, modelData.WeightsHiddenInput.shape);
            Tensor loadedWeightsInputCell = new Tensor(modelData.WeightsInputCell.data, modelData.WeightsInputCell.shape);
            Tensor loadedWeightsHiddenCell = new Tensor(modelData.WeightsHiddenCell.data, modelData.WeightsHiddenCell.shape);
            Tensor loadedWeightsInputOutput = new Tensor(modelData.WeightsInputOutput.data, modelData.WeightsInputOutput.shape);
            Tensor loadedWeightsHiddenOutput = new Tensor(modelData.WeightsHiddenOutput.data, modelData.WeightsHiddenOutput.shape);
            Tensor loadedBiasForget = new Tensor(modelData.BiasForget.data, modelData.BiasForget.shape);
            Tensor loadedBiasInput = new Tensor(modelData.BiasInput.data, modelData.BiasInput.shape);
            Tensor loadedBiasCell = new Tensor(modelData.BiasCell.data, modelData.BiasCell.shape);
            Tensor loadedBiasOutput = new Tensor(modelData.BiasOutput.data, modelData.BiasOutput.shape);
            Tensor loadedWeightsHiddenOutputFinal = new Tensor(modelData.WeightsHiddenOutputFinal.data, modelData.WeightsHiddenOutputFinal.shape);
            Tensor loadedBiasOutputFinal = new Tensor(modelData.BiasOutputFinal.data, modelData.BiasOutputFinal.shape);
            
            return new NeuralNetworkLSTM(modelData.InputSize, modelData.HiddenSize, modelData.OutputSize,
                                        loadedWeightsInputForget, loadedWeightsHiddenForget,
                                        loadedWeightsInputInput, loadedWeightsHiddenInput,
                                        loadedWeightsInputCell, loadedWeightsHiddenCell,
                                        loadedWeightsInputOutput, loadedWeightsHiddenOutput,
                                        loadedBiasForget, loadedBiasInput, loadedBiasCell, loadedBiasOutput,
                                        loadedWeightsHiddenOutputFinal, loadedBiasOutputFinal,
                                        openCLService);
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Erro ao carregar o modelo LSTM: {ex.Message}");
            return null;
        }
    }
}