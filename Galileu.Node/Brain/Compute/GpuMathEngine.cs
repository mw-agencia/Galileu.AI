using OpenCL.Net;
using System.Runtime.InteropServices;
using Galileu.Node.Core;
using Galileu.Node.Interfaces;

namespace Galileu.Node.Brain.Compute;

public class GpuMathEngine : IMathEngine
{
    private class GpuTensor : IMathTensor
    {
        public IMem Buffer { get; }
        public int[] Shape { get; }
        public int TotalSize => Shape.Aggregate(1, (acc, val) => acc * val);
        public GpuTensor(IMem buffer, int[] shape) { Buffer = buffer; Shape = shape; }
        public void Dispose() => Buffer?.Dispose();
    }

    public bool IsGpu => true;
    private readonly Context _context;
    private readonly CommandQueue _queue;
    private readonly Program _program;
    private readonly Kernel _matmulKernel, _addKernel, _addBroadcastKernel, _multiplyKernel, _sigmoidKernel, _tanhKernel;

    public GpuMathEngine()
    {
        var platform = Cl.GetPlatformIDs().First();
        var device = Cl.GetDeviceIDs(platform, DeviceType.Gpu).First();
        _context = Cl.CreateContext(null, 1, new[] { device }, null, IntPtr.Zero);
        _queue = Cl.CreateCommandQueue(_context, device, CommandQueueProperties.None);
        
        var kernelSource = File.ReadAllText(Path.Combine(AppContext.BaseDirectory, "Kernels", "MatrixOperations.cl"));
        _program = Cl.CreateProgramWithSource(_context, 1, new[] { kernelSource }, null);
        Cl.BuildProgram(_program, 1, new[] { device }, string.Empty, null, IntPtr.Zero);

        if (Cl.GetProgramBuildInfo(_program, device, ProgramBuildInfo.Status, out var buildStatus).IsSuccess &&
            (BuildStatus)buildStatus.CastTo<int>() != BuildStatus.Success)
        {
            Cl.GetProgramBuildInfo(_program, device, ProgramBuildInfo.Log, out var log);
            throw new Exception("Erro de compilação do Kernel OpenCL: " + log.CastTo<string>());
        }

        _matmulKernel = Cl.CreateKernel(_program, "matmul_forward");
        _addKernel = Cl.CreateKernel(_program, "elementwise_add_forward");
        _addBroadcastKernel = Cl.CreateKernel(_program, "elementwise_add_broadcast_forward");
        _multiplyKernel = Cl.CreateKernel(_program, "elementwise_multiply");
        _sigmoidKernel = Cl.CreateKernel(_program, "sigmoid_forward");
        _tanhKernel = Cl.CreateKernel(_program, "tanh_forward");
    }

    public IMathTensor CreateTensor(double[] hostData, int[] shape)
    {
        var buffer = Cl.CreateBuffer(_context, MemFlags.ReadWrite | MemFlags.CopyHostPtr, hostData, GCHandle.Alloc(hostData, GCHandleType.Pinned));
        return new GpuTensor(buffer, shape);
    }
    
    public Tensor ReadTensor(IMathTensor tensor)
    {
        var gpuTensor = (GpuTensor)tensor;
        var data = new double[gpuTensor.TotalSize];
        Cl.EnqueueReadBuffer(_queue, gpuTensor.Buffer, Bool.True, 0, (long)gpuTensor.TotalSize * sizeof(double), data, 0, null, out _);
        return new Tensor(data, gpuTensor.Shape);
    }
    
    // Implementações que chamam os kernels (código omitido por brevidade, mas segue o padrão da resposta anterior)
    public void MatrixMultiply(IMathTensor a, IMathTensor b, IMathTensor result)
    {
        int M = a.Shape[0]; int K = a.Shape[1]; int N = b.Shape[1];
        Cl.SetKernelArg(_matmulKernel, 0, ((GpuTensor)a).Buffer); Cl.SetKernelArg(_matmulKernel, 1, ((GpuTensor)b).Buffer); Cl.SetKernelArg(_matmulKernel, 2, ((GpuTensor)result).Buffer);
        Cl.SetKernelArg(_matmulKernel, 3, (uint)M); Cl.SetKernelArg(_matmulKernel, 4, (uint)K); Cl.SetKernelArg(_matmulKernel, 5, (uint)N);
        Cl.EnqueueNDRangeKernel(_queue, _matmulKernel, 2, null, new[] { (IntPtr)N, (IntPtr)M }, null, 0, null, out _);
    }
    public void Add(IMathTensor a, IMathTensor b, IMathTensor result)
    {
        Cl.SetKernelArg(_addKernel, 0, ((GpuTensor)a).Buffer); Cl.SetKernelArg(_addKernel, 1, ((GpuTensor)b).Buffer); Cl.SetKernelArg(_addKernel, 2, ((GpuTensor)result).Buffer);
        Cl.SetKernelArg(_addKernel, 3, (uint)((GpuTensor)a).TotalSize);
        Cl.EnqueueNDRangeKernel(_queue, _addKernel, 1, null, new[] { (IntPtr)((GpuTensor)a).TotalSize }, null, 0, null, out _);
    }
    public void AddBroadcast(IMathTensor matrix, IMathTensor vector, IMathTensor result)
    {
        int M = matrix.Shape[0]; int N = matrix.Shape[1];
        Cl.SetKernelArg(_addBroadcastKernel, 0, ((GpuTensor)matrix).Buffer); Cl.SetKernelArg(_addBroadcastKernel, 1, ((GpuTensor)vector).Buffer); Cl.SetKernelArg(_addBroadcastKernel, 2, ((GpuTensor)result).Buffer);
        Cl.SetKernelArg(_addBroadcastKernel, 3, (uint)M); Cl.SetKernelArg(_addBroadcastKernel, 4, (uint)N);
        Cl.EnqueueNDRangeKernel(_queue, _addBroadcastKernel, 2, null, new[] { (IntPtr)N, (IntPtr)M }, null, 0, null, out _);
    }
    public void Multiply(IMathTensor a, IMathTensor b, IMathTensor result)
    {
        Cl.SetKernelArg(_multiplyKernel, 0, ((GpuTensor)a).Buffer); Cl.SetKernelArg(_multiplyKernel, 1, ((GpuTensor)b).Buffer); Cl.SetKernelArg(_multiplyKernel, 2, ((GpuTensor)result).Buffer);
        Cl.SetKernelArg(_multiplyKernel, 3, (uint)((GpuTensor)a).TotalSize);
        Cl.EnqueueNDRangeKernel(_queue, _multiplyKernel, 1, null, new[] { (IntPtr)((GpuTensor)a).TotalSize }, null, 0, null, out _);
    }
    public void Sigmoid(IMathTensor input, IMathTensor result)
    {
        Cl.SetKernelArg(_sigmoidKernel, 0, ((GpuTensor)input).Buffer); Cl.SetKernelArg(_sigmoidKernel, 1, ((GpuTensor)result).Buffer);
        Cl.SetKernelArg(_sigmoidKernel, 2, (uint)((GpuTensor)input).TotalSize);
        Cl.EnqueueNDRangeKernel(_queue, _sigmoidKernel, 1, null, new[] { (IntPtr)((GpuTensor)input).TotalSize }, null, 0, null, out _);
    }
    public void Tanh(IMathTensor input, IMathTensor result)
    {
        Cl.SetKernelArg(_tanhKernel, 0, ((GpuTensor)input).Buffer); Cl.SetKernelArg(_tanhKernel, 1, ((GpuTensor)result).Buffer);
        Cl.SetKernelArg(_tanhKernel, 2, (uint)((GpuTensor)input).TotalSize);
        Cl.EnqueueNDRangeKernel(_queue, _tanhKernel, 1, null, new[] { (IntPtr)((GpuTensor)input).TotalSize }, null, 0, null, out _);
    }

    public void Dispose()
    {
        _matmulKernel?.Dispose(); _addKernel?.Dispose(); _addBroadcastKernel?.Dispose(); 
        _multiplyKernel?.Dispose(); _sigmoidKernel?.Dispose(); _tanhKernel?.Dispose();
        _program?.Dispose(); _queue?.Dispose(); _context?.Dispose();
    }
}