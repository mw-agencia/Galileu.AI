using Galileu.Node.Core;
using Galileu.Node.Interfaces;
using OpenCL.NetCore;
using System;
using static OpenCL.NetCore.Cl;

namespace Galileu.Node.Gpu;

public class GpuMathEngine : IMathEngine
{
    private int _operationsWithoutSync = 0;
    private const int SYNC_INTERVAL = 16;
    public bool IsGpu => true;

    private readonly Context _context;
    private readonly CommandQueue _commandQueue;
    private readonly OpenCL.NetCore.Program _program;

    private readonly Kernel _matrixMultiplyKernel;
    private readonly Kernel _addKernel;
    private readonly Kernel _addBroadcastKernel;
    private readonly Kernel _multiplyKernel;
    private readonly Kernel _sigmoidKernel;
    private readonly Kernel _tanhKernel;
    private readonly Kernel _cloneKernel;
    private readonly Kernel _transposeKernel;
    private readonly Kernel _subtractKernel;
    private readonly Kernel _sigmoidDerivativeKernel;
    private readonly Kernel _tanhDerivativeKernel;
    private readonly Kernel _matrixMultiplyTransposeAKernel;
    private readonly Kernel _matrixMultiplyTransposeBKernel;
    private readonly Kernel _addScaledKernel;
    private readonly Kernel _subtractScaledKernel;
    private readonly Kernel _sliceKernel;
    private readonly Kernel _setKernel;
    private readonly Kernel _clipKernel;
    private readonly Kernel _scaleKernel;
    private readonly Kernel _softmaxKernel;
    private readonly Kernel _lookupKernel;
    private readonly Kernel _accumulateGradientKernel;
    private readonly Kernel _softmaxCrossEntropyGradientKernel;

    private bool _disposed = false;

    #region Kernels OpenCL
    
    public void Synchronize()
    {
        ErrorCode error = Finish(_commandQueue);
        if (error != ErrorCode.Success)
        {
            Console.WriteLine($"[GPU] Aviso: Erro na sincronização: {error}");
        }
        _operationsWithoutSync = 0;
    }

    private void MaybeSynchronize()
    {
        _operationsWithoutSync++;
        if (_operationsWithoutSync >= SYNC_INTERVAL)
        {
            Synchronize();
        }
    }

    private const string ProgramSource = @"
            // KERNELS DO FORWARD PASS
            __kernel void matrix_multiply(__global const float* A, __global const float* B, __global float* C, int M, int N, int P) { int i = get_global_id(0); int j = get_global_id(1); if (i < M && j < P) { float sum = 0.0f; for (int k = 0; k < N; ++k) { sum += A[i * N + k] * B[k * P + j]; } C[i * P + j] = sum; } }
            __kernel void add(__global const float* a, __global const float* b, __global float* result) { int gid = get_global_id(0); result[gid] = a[gid] + b[gid]; }
            __kernel void add_broadcast(__global float* a, __global const float* bias, int bias_size) { int gid = get_global_id(0); a[gid] += bias[gid % bias_size]; }
            __kernel void multiply(__global const float* a, __global const float* b, __global float* result) { int gid = get_global_id(0); result[gid] = a[gid] * b[gid]; }
            __kernel void sigmoid(__global const float* a, __global float* result) { int gid = get_global_id(0); result[gid] = 1.0f / (1.0f + exp(-a[gid])); }
            __kernel void tanh_activation(__global const float* a, __global float* result) { int gid = get_global_id(0); result[gid] = tanh(a[gid]); }
            
            // KERNELS DO BACKWARD PASS (BPTT) E UTILITÁRIOS
            __kernel void clone_buffer(__global const float* input, __global float* output) { int gid = get_global_id(0); output[gid] = input[gid]; }
            __kernel void transpose(__global const float* input, __global float* output, int rows, int cols) { int i = get_global_id(0); int j = get_global_id(1); if (i < rows && j < cols) { output[j * rows + i] = input[i * cols + j]; } }
            __kernel void subtract(__global const float* a, __global const float* b, __global float* result) { int gid = get_global_id(0); result[gid] = a[gid] - b[gid]; }
            __kernel void sigmoid_derivative(__global const float* output, __global float* result) { int gid = get_global_id(0); float o = output[gid]; result[gid] = o * (1.0f - o); }
            __kernel void tanh_derivative(__global const float* output, __global float* result) { int gid = get_global_id(0); float o = output[gid]; result[gid] = 1.0f - o * o; }
            __kernel void matrix_multiply_transpose_a(__global const float* A, __global const float* B, __global float* C, int M, int K, int P) { int i = get_global_id(0); int j = get_global_id(1); if (i < M && j < P) { float sum = 0.0f; for (int k = 0; k < K; ++k) { sum += A[k * M + i] * B[k * P + j]; } C[i * P + j] = sum; } }
            __kernel void matrix_multiply_transpose_b(__global const float* A, __global const float* B, __global float* C, int M, int K, int P) { int i = get_global_id(0); int j = get_global_id(1); if (i < M && j < P) { float sum = 0.0f; for (int k = 0; k < K; ++k) { sum += A[i * K + k] * B[j * K + k]; } C[i * P + j] = sum; } }
            __kernel void add_scaled(__global float* target, __global const float* source, float scalar) { int gid = get_global_id(0); target[gid] += source[gid] * scalar; }
            __kernel void subtract_scaled(__global float* target, __global const float* source, float scalar) { int gid = get_global_id(0); target[gid] -= source[gid] * scalar; }
            __kernel void slice(__global const float* source, __global float* dest, int offset, int size) { int gid = get_global_id(0); if (gid < size) { dest[gid] = source[offset + gid]; } }
            __kernel void set(__global float* dest, __global const float* source, int offset, int size) { int gid = get_global_id(0); if (gid < size) { dest[offset + gid] = source[gid]; } }
            __kernel void clip(__global float* data, float min_val, float max_val) { int gid = get_global_id(0); data[gid] = fmax(min_val, fmin(max_val, data[gid])); }
            __kernel void scale(__global float* data, float scalar) { int gid = get_global_id(0); data[gid] *= scalar; }
            __kernel void softmax(__global const float* input, __global float* output, int size) { int row = get_global_id(0); int offset = row * size; float maxVal = input[offset]; for (int i = 1; i < size; i++) { float val = input[offset + i]; if (val > maxVal) maxVal = val; } float sumExp = 0.0f; for (int i = 0; i < size; i++) { output[offset + i] = exp(input[offset + i] - maxVal); sumExp += output[offset + i]; } for (int i = 0; i < size; i++) { output[offset + i] /= sumExp; } }
            __kernel void lookup(__global const float* embedding_matrix, __global float* result, int index, int embedding_size) { int gid = get_global_id(0); if (gid < embedding_size) { result[gid] = embedding_matrix[index * embedding_size + gid]; } }   
            __kernel void accumulate_gradient_no_atomic(__global float* embedding_gradients, __global const float* gradient, int index, int embedding_size) { int gid = get_global_id(0); if (gid < embedding_size) { embedding_gradients[index * embedding_size + gid] += gradient[gid]; } }
            __kernel void softmax_cross_entropy_gradient(__global float* data, __global const int* targets, int vocab_size) { int t = get_global_id(0); int target_idx = targets[t]; int flat_idx = t * vocab_size + target_idx; data[flat_idx] -= 1.0f; }
        ";

    #endregion

    public GpuMathEngine()
    {
        ErrorCode error;
        Platform[] platforms = GetPlatformIDs(out error);
        CheckError(error, "Erro ao obter plataformas OpenCL.");
        if (platforms.Length == 0) throw new InvalidOperationException("Nenhuma plataforma OpenCL encontrada.");
        var platform = platforms.First();
        Device[] devices = GetDeviceIDs(platform, DeviceType.Gpu, out error);
        if (error != ErrorCode.Success || devices.Length == 0)
        {
            devices = GetDeviceIDs(platform, DeviceType.Cpu, out error);
            CheckError(error, "Nenhum dispositivo OpenCL (GPU ou CPU) encontrado.");
        }
        var device = devices[0];
        var deviceName = GetDeviceInfo(device, DeviceInfo.Name, out error).ToString();
        Console.WriteLine($"[OpenCL] Usando dispositivo: {deviceName}");
        _context = CreateContext(null, 1, new[] { device }, null, IntPtr.Zero, out error);
        CheckError(error);
        _commandQueue = CreateCommandQueue(_context, device, CommandQueueProperties.None, out error);
        CheckError(error);
        _program = CreateProgramWithSource(_context, 1, new[] { ProgramSource }, null, out error);
        CheckError(error);
        error = BuildProgram(_program, 1, new[] { device }, string.Empty, null, IntPtr.Zero);
        if (error != ErrorCode.Success)
        {
            var buildLog = GetProgramBuildInfo(_program, device, ProgramBuildInfo.Log, out _).ToString();
            throw new OpenClException($"Erro ao compilar kernels OpenCL: {buildLog}", error);
        }

        _matrixMultiplyKernel = CreateKernel(_program, "matrix_multiply", out error); CheckError(error);
        _addKernel = CreateKernel(_program, "add", out error); CheckError(error);
        _addBroadcastKernel = CreateKernel(_program, "add_broadcast", out error); CheckError(error);
        _multiplyKernel = CreateKernel(_program, "multiply", out error); CheckError(error);
        _sigmoidKernel = CreateKernel(_program, "sigmoid", out error); CheckError(error);
        _tanhKernel = CreateKernel(_program, "tanh_activation", out error); CheckError(error);
        _cloneKernel = CreateKernel(_program, "clone_buffer", out error); CheckError(error);
        _transposeKernel = CreateKernel(_program, "transpose", out error); CheckError(error);
        _subtractKernel = CreateKernel(_program, "subtract", out error); CheckError(error);
        _sigmoidDerivativeKernel = CreateKernel(_program, "sigmoid_derivative", out error); CheckError(error);
        _tanhDerivativeKernel = CreateKernel(_program, "tanh_derivative", out error); CheckError(error);
        _matrixMultiplyTransposeAKernel = CreateKernel(_program, "matrix_multiply_transpose_a", out error); CheckError(error);
        _matrixMultiplyTransposeBKernel = CreateKernel(_program, "matrix_multiply_transpose_b", out error); CheckError(error);
        _addScaledKernel = CreateKernel(_program, "add_scaled", out error); CheckError(error);
        _subtractScaledKernel = CreateKernel(_program, "subtract_scaled", out error); CheckError(error);
        _sliceKernel = CreateKernel(_program, "slice", out error); CheckError(error);
        _setKernel = CreateKernel(_program, "set", out error); CheckError(error);
        _clipKernel = CreateKernel(_program, "clip", out error); CheckError(error);
        _scaleKernel = CreateKernel(_program, "scale", out error); CheckError(error);
        _softmaxKernel = CreateKernel(_program, "softmax", out error); CheckError(error);
        _lookupKernel = CreateKernel(_program, "lookup", out error); CheckError(error);
        _accumulateGradientKernel = CreateKernel(_program, "accumulate_gradient_no_atomic", out error); CheckError(error);
        _softmaxCrossEntropyGradientKernel = CreateKernel(_program, "softmax_cross_entropy_gradient", out error); CheckError(error);
    }

    public IMathTensor CreateTensor(int[] shape) => new GpuTensor(shape, _context, _commandQueue);
    public IMathTensor CreateTensor(double[] data, int[] shape) => new GpuTensor(data, shape, _context, _commandQueue);

    public void MatrixMultiply(IMathTensor a, IMathTensor b, IMathTensor result)
    {
        var tensorA = (GpuTensor)a;
        var tensorB = (GpuTensor)b;
        var tensorC = (GpuTensor)result;
        int M = tensorA.Shape[0];
        int N = tensorA.Shape[1];
        int P = tensorB.Shape[1];
        SetKernelArg(_matrixMultiplyKernel, 0, tensorA.Buffer);
        SetKernelArg(_matrixMultiplyKernel, 1, tensorB.Buffer);
        SetKernelArg(_matrixMultiplyKernel, 2, tensorC.Buffer);
        SetKernelArg(_matrixMultiplyKernel, 3, (uint)M);
        SetKernelArg(_matrixMultiplyKernel, 4, (uint)N);
        SetKernelArg(_matrixMultiplyKernel, 5, (uint)P);
        EnqueueNDRangeKernel(_commandQueue, _matrixMultiplyKernel, 2, null, new[] { (IntPtr)M, (IntPtr)P }, null, 0, null, out _);
        MaybeSynchronize();
    }

    public void Add(IMathTensor a, IMathTensor b, IMathTensor result)
    {
        SetKernelArg(_addKernel, 0, ((GpuTensor)a).Buffer);
        SetKernelArg(_addKernel, 1, ((GpuTensor)b).Buffer);
        SetKernelArg(_addKernel, 2, ((GpuTensor)result).Buffer);
        EnqueueNDRangeKernel(_commandQueue, _addKernel, 1, null, new[] { (IntPtr)a.Length }, null, 0, null, out _);
    }

    public void AddBroadcast(IMathTensor a, IMathTensor bias, IMathTensor result)
    {
        SetKernelArg(_addBroadcastKernel, 0, ((GpuTensor)a).Buffer);
        SetKernelArg(_addBroadcastKernel, 1, ((GpuTensor)bias).Buffer);
        SetKernelArg(_addBroadcastKernel, 2, (uint)bias.Length);
        EnqueueNDRangeKernel(_commandQueue, _addBroadcastKernel, 1, null, new[] { (IntPtr)a.Length }, null, 0, null, out _);
    }

    public void Multiply(IMathTensor a, IMathTensor b, IMathTensor result)
    {
        SetKernelArg(_multiplyKernel, 0, ((GpuTensor)a).Buffer);
        SetKernelArg(_multiplyKernel, 1, ((GpuTensor)b).Buffer);
        SetKernelArg(_multiplyKernel, 2, ((GpuTensor)result).Buffer);
        EnqueueNDRangeKernel(_commandQueue, _multiplyKernel, 1, null, new[] { (IntPtr)a.Length }, null, 0, null, out _);
    }

    public void Sigmoid(IMathTensor a, IMathTensor result)
    {
        SetKernelArg(_sigmoidKernel, 0, ((GpuTensor)a).Buffer);
        SetKernelArg(_sigmoidKernel, 1, ((GpuTensor)result).Buffer);
        EnqueueNDRangeKernel(_commandQueue, _sigmoidKernel, 1, null, new[] { (IntPtr)a.Length }, null, 0, null, out _);
    }

    public void Tanh(IMathTensor a, IMathTensor result)
    {
        SetKernelArg(_tanhKernel, 0, ((GpuTensor)a).Buffer);
        SetKernelArg(_tanhKernel, 1, ((GpuTensor)result).Buffer);
        EnqueueNDRangeKernel(_commandQueue, _tanhKernel, 1, null, new[] { (IntPtr)a.Length }, null, 0, null, out _);
    }

    public IMathTensor Clone(IMathTensor tensor)
    {
        var gpuTensor = (GpuTensor)tensor;
        var newTensor = CreateTensor(gpuTensor.Shape) as GpuTensor;
        Copy(gpuTensor, newTensor!);
        return newTensor;
    }

    public void Transpose(IMathTensor input, IMathTensor result)
    {
        var tensorIn = (GpuTensor)input;
        var tensorOut = (GpuTensor)result;
        int rows = tensorIn.Shape[0];
        int cols = tensorIn.Shape[1];
        SetKernelArg(_transposeKernel, 0, tensorIn.Buffer);
        SetKernelArg(_transposeKernel, 1, tensorOut.Buffer);
        SetKernelArg(_transposeKernel, 2, (uint)rows);
        SetKernelArg(_transposeKernel, 3, (uint)cols);
        EnqueueNDRangeKernel(_commandQueue, _transposeKernel, 2, null, new[] { (IntPtr)rows, (IntPtr)cols }, null, 0, null, out _);
    }

    public void Subtract(IMathTensor a, IMathTensor b, IMathTensor result)
    {
        SetKernelArg(_subtractKernel, 0, ((GpuTensor)a).Buffer);
        SetKernelArg(_subtractKernel, 1, ((GpuTensor)b).Buffer);
        SetKernelArg(_subtractKernel, 2, ((GpuTensor)result).Buffer);
        EnqueueNDRangeKernel(_commandQueue, _subtractKernel, 1, null, new[] { (IntPtr)a.Length }, null, 0, null, out _);
    }

    public void SigmoidDerivative(IMathTensor output, IMathTensor result)
    {
        SetKernelArg(_sigmoidDerivativeKernel, 0, ((GpuTensor)output).Buffer);
        SetKernelArg(_sigmoidDerivativeKernel, 1, ((GpuTensor)result).Buffer);
        EnqueueNDRangeKernel(_commandQueue, _sigmoidDerivativeKernel, 1, null, new[] { (IntPtr)output.Length }, null, 0, null, out _);
    }

    public void TanhDerivative(IMathTensor output, IMathTensor result)
    {
        SetKernelArg(_tanhDerivativeKernel, 0, ((GpuTensor)output).Buffer);
        SetKernelArg(_tanhDerivativeKernel, 1, ((GpuTensor)result).Buffer);
        EnqueueNDRangeKernel(_commandQueue, _tanhDerivativeKernel, 1, null, new[] { (IntPtr)output.Length }, null, 0, null, out _);
    }

    public void MatrixMultiplyTransposeA(IMathTensor a, IMathTensor b, IMathTensor result)
    {
        var tensorA = (GpuTensor)a;
        var tensorB = (GpuTensor)b;
        var tensorC = (GpuTensor)result;
        int M = tensorA.Shape[1];
        int K = tensorA.Shape[0];
        int P = tensorB.Shape[1];
        SetKernelArg(_matrixMultiplyTransposeAKernel, 0, tensorA.Buffer);
        SetKernelArg(_matrixMultiplyTransposeAKernel, 1, tensorB.Buffer);
        SetKernelArg(_matrixMultiplyTransposeAKernel, 2, tensorC.Buffer);
        SetKernelArg(_matrixMultiplyTransposeAKernel, 3, (uint)M);
        SetKernelArg(_matrixMultiplyTransposeAKernel, 4, (uint)K);
        SetKernelArg(_matrixMultiplyTransposeAKernel, 5, (uint)P);
        EnqueueNDRangeKernel(_commandQueue, _matrixMultiplyTransposeAKernel, 2, null, new[] { (IntPtr)M, (IntPtr)P }, null, 0, null, out _);
    }

    public void MatrixMultiplyTransposeB(IMathTensor a, IMathTensor b, IMathTensor result)
    {
        var tensorA = (GpuTensor)a;
        var tensorB = (GpuTensor)b;
        var tensorC = (GpuTensor)result;
        int M = tensorA.Shape[0];
        int K = tensorA.Shape[1];
        int P = tensorB.Shape[0];
        SetKernelArg(_matrixMultiplyTransposeBKernel, 0, tensorA.Buffer);
        SetKernelArg(_matrixMultiplyTransposeBKernel, 1, tensorB.Buffer);
        SetKernelArg(_matrixMultiplyTransposeBKernel, 2, tensorC.Buffer);
        SetKernelArg(_matrixMultiplyTransposeBKernel, 3, (uint)M);
        SetKernelArg(_matrixMultiplyTransposeBKernel, 4, (uint)K);
        SetKernelArg(_matrixMultiplyTransposeBKernel, 5, (uint)P);
        EnqueueNDRangeKernel(_commandQueue, _matrixMultiplyTransposeBKernel, 2, null, new[] { (IntPtr)M, (IntPtr)P }, null, 0, null, out _);
    }

    public void AddScaled(IMathTensor target, IMathTensor source, double scalar)
    {
        SetKernelArg(_addScaledKernel, 0, ((GpuTensor)target).Buffer);
        SetKernelArg(_addScaledKernel, 1, ((GpuTensor)source).Buffer);
        SetKernelArg(_addScaledKernel, 2, (float)scalar);
        EnqueueNDRangeKernel(_commandQueue, _addScaledKernel, 1, null, new[] { (IntPtr)target.Length }, null, 0, null, out _);
    }

    public void SubtractScaled(IMathTensor target, IMathTensor source, double scalar)
    {
        SetKernelArg(_subtractScaledKernel, 0, ((GpuTensor)target).Buffer);
        SetKernelArg(_subtractScaledKernel, 1, ((GpuTensor)source).Buffer);
        SetKernelArg(_subtractScaledKernel, 2, (float)scalar);
        EnqueueNDRangeKernel(_commandQueue, _subtractScaledKernel, 1, null, new[] { (IntPtr)target.Length }, null, 0, null, out _);
    }

    public void Slice(IMathTensor source, int rowIndex, IMathTensor destination)
    {
        var featureSize = (int)destination.Length;
        var offset = rowIndex * featureSize;
        SetKernelArg(_sliceKernel, 0, ((GpuTensor)source).Buffer);
        SetKernelArg(_sliceKernel, 1, ((GpuTensor)destination).Buffer);
        SetKernelArg(_sliceKernel, 2, (uint)offset);
        SetKernelArg(_sliceKernel, 3, (uint)featureSize);
        EnqueueNDRangeKernel(_commandQueue, _sliceKernel, 1, null, new[] { (IntPtr)featureSize }, null, 0, null, out _);
    }

    public void Set(IMathTensor destination, int rowIndex, IMathTensor source)
    {
        var featureSize = (int)source.Length;
        var offset = rowIndex * featureSize;
        SetKernelArg(_setKernel, 0, ((GpuTensor)destination).Buffer);
        SetKernelArg(_setKernel, 1, ((GpuTensor)source).Buffer);
        SetKernelArg(_setKernel, 2, (uint)offset);
        SetKernelArg(_setKernel, 3, (uint)featureSize);
        EnqueueNDRangeKernel(_commandQueue, _setKernel, 1, null, new[] { (IntPtr)featureSize }, null, 0, null, out _);
    }

    public void Clip(IMathTensor tensor, double minValue, double maxValue)
    {
        SetKernelArg(_clipKernel, 0, ((GpuTensor)tensor).Buffer);
        SetKernelArg(_clipKernel, 1, (float)minValue);
        SetKernelArg(_clipKernel, 2, (float)maxValue);
        EnqueueNDRangeKernel(_commandQueue, _clipKernel, 1, null, new[] { (IntPtr)tensor.Length }, null, 0, null, out _);
    }
    
    public void Scale(IMathTensor tensor, double scalar)
    {
        SetKernelArg(_scaleKernel, 0, ((GpuTensor)tensor).Buffer);
        SetKernelArg(_scaleKernel, 1, (float)scalar);
        EnqueueNDRangeKernel(_commandQueue, _scaleKernel, 1, null, new[] { (IntPtr)tensor.Length }, null, 0, null, out _);
    }
    
    public void Softmax(IMathTensor input, IMathTensor result)
    {
        var tensorIn = (GpuTensor)input;
        var tensorOut = (GpuTensor)result;
        int rows = tensorIn.Shape[0];
        int cols = tensorIn.Shape[1];
        SetKernelArg(_softmaxKernel, 0, tensorIn.Buffer);
        SetKernelArg(_softmaxKernel, 1, tensorOut.Buffer);
        SetKernelArg(_softmaxKernel, 2, (uint)cols);
        EnqueueNDRangeKernel(_commandQueue, _softmaxKernel, 1, null, new[] { (IntPtr)rows }, null, 0, null, out _);
    }
    
    public void Lookup(IMathTensor embeddingMatrix, int index, IMathTensor result)
    {
        var embeddingSize = embeddingMatrix.Shape[1];
        SetKernelArg(_lookupKernel, 0, ((GpuTensor)embeddingMatrix).Buffer);
        SetKernelArg(_lookupKernel, 1, ((GpuTensor)result).Buffer);
        SetKernelArg(_lookupKernel, 2, (uint)index);
        SetKernelArg(_lookupKernel, 3, (uint)embeddingSize);
        EnqueueNDRangeKernel(_commandQueue, _lookupKernel, 1, null, new[] { (IntPtr)embeddingSize }, null, 0, null, out _);
    }

    public void AccumulateGradient(IMathTensor embeddingGradients, IMathTensor gradient, int index)
    {
        var embeddingSize = embeddingGradients.Shape[1];
        SetKernelArg(_accumulateGradientKernel, 0, ((GpuTensor)embeddingGradients).Buffer);
        SetKernelArg(_accumulateGradientKernel, 1, ((GpuTensor)gradient).Buffer);
        SetKernelArg(_accumulateGradientKernel, 2, (uint)index);
        SetKernelArg(_accumulateGradientKernel, 3, (uint)embeddingSize);
        EnqueueNDRangeKernel(_commandQueue, _accumulateGradientKernel, 1, null, new[] { (IntPtr)embeddingSize }, null, 0, null, out _);
    }
    
    public void SoftmaxCrossEntropyGradient(IMathTensor predictions, int[] targetIndices, IMathTensor result)
    {
        this.Copy(predictions, result);

        ErrorCode error;
        var targetsGpuBuffer = (Mem)CreateBuffer(_context, MemFlags.ReadOnly | MemFlags.CopyHostPtr, 
            (IntPtr)(targetIndices.Length * sizeof(int)), targetIndices, out error);
        CheckError(error);

        try
        {
            SetKernelArg(_softmaxCrossEntropyGradientKernel, 0, ((GpuTensor)result).Buffer);
            SetKernelArg(_softmaxCrossEntropyGradientKernel, 1, targetsGpuBuffer);
            SetKernelArg(_softmaxCrossEntropyGradientKernel, 2, (uint)predictions.Shape[1]);

            EnqueueNDRangeKernel(_commandQueue, _softmaxCrossEntropyGradientKernel, 1, null, 
                new[] { (IntPtr)targetIndices.Length }, null, 0, null, out _);
        }
        finally
        {
            ReleaseMemObject(targetsGpuBuffer);
        }
    }
    public void Copy(IMathTensor source, IMathTensor destination)
    {
        if (source.Length != destination.Length)
            throw new ArgumentException("Os tensores de origem e destino devem ter o mesmo tamanho para a cópia.");

        var gpuSource = (GpuTensor)source;
        var gpuDestination = (GpuTensor)destination;
        
        SetKernelArg(_cloneKernel, 0, gpuSource.Buffer);
        SetKernelArg(_cloneKernel, 1, gpuDestination.Buffer);
        
        EnqueueNDRangeKernel(_commandQueue, _cloneKernel, 1, null, new[] { (IntPtr)gpuSource.Length }, null, 0, null, out _);
    }

    public void Dispose()
    {
        if (_disposed) return;
    
        try
        {
            Synchronize();
        }
        catch (System.Exception ex)
        {
            Console.WriteLine($"[GPU] Erro na sincronização final: {ex.Message}");
        }

        ReleaseKernel(_matrixMultiplyKernel);
        ReleaseKernel(_addKernel);
        ReleaseKernel(_addBroadcastKernel);
        ReleaseKernel(_multiplyKernel);
        ReleaseKernel(_sigmoidKernel);
        ReleaseKernel(_tanhKernel);
        ReleaseKernel(_cloneKernel);
        ReleaseKernel(_transposeKernel);
        ReleaseKernel(_subtractKernel);
        ReleaseKernel(_sigmoidDerivativeKernel);
        ReleaseKernel(_tanhDerivativeKernel);
        ReleaseKernel(_matrixMultiplyTransposeAKernel);
        ReleaseKernel(_matrixMultiplyTransposeBKernel);
        ReleaseKernel(_addScaledKernel);
        ReleaseKernel(_subtractScaledKernel);
        ReleaseKernel(_sliceKernel);
        ReleaseKernel(_setKernel);
        ReleaseKernel(_clipKernel);
        ReleaseKernel(_scaleKernel);
        ReleaseKernel(_softmaxKernel);
        ReleaseKernel(_lookupKernel);
        ReleaseKernel(_accumulateGradientKernel);
        ReleaseKernel(_softmaxCrossEntropyGradientKernel);

        ReleaseProgram(_program);
        ReleaseCommandQueue(_commandQueue);
        ReleaseContext(_context);
        
        _disposed = true;
        GC.SuppressFinalize(this);
    }

    private void CheckError(ErrorCode error, string message = "Erro OpenCL não especificado.")
    {
        if (error != ErrorCode.Success) throw new OpenClException(message, error);
    }
}