using Galileu.Node.Interfaces;
using OpenCL.Net;

namespace Galileu.Node.Brain.Gpu;

public class GpuMathEngine : IMathEngine
{
    private readonly OpenCLService _service;
    public bool IsGpu => true;

    public GpuMathEngine(OpenCLService service)
    {
        _service = service ?? throw new ArgumentNullException(nameof(service));
        if (!_service.IsGpuAvailable)
        {
            throw new InvalidOperationException("GPU não está disponível para criar um GpuMathEngine.");
        }
    }

    public IMathTensor CreateTensor(int[] shape)
    {
        return new GpuTensor(_service, shape);
    }

    public IMathTensor CreateTensor(double[] hostData, int[] shape)
    {
        var tensor = new GpuTensor(_service, shape);
        tensor.Write(hostData);
        return tensor;
    }

    public void MatrixMultiply(IMathTensor a, IMathTensor b, IMathTensor result)
    {
        if (a is not GpuTensor gA || b is not GpuTensor gB || result is not GpuTensor gC)
            throw new ArgumentException("Os tensores devem ser GpuTensors.");

        int M = gA.Shape[0];
        int K = gA.Shape[1];
        int N = gB.Shape[1];

        Kernel kernel = _service.Kernels["matmul_forward"];
        Cl.SetKernelArg(kernel, 0, (IntPtr) IntPtr.Size, gA.Buffer);
        Cl.SetKernelArg(kernel, 1, (IntPtr) IntPtr.Size, gB.Buffer);
        Cl.SetKernelArg(kernel, 2, (IntPtr) IntPtr.Size, gC.Buffer);
        Cl.SetKernelArg(kernel, 3, (IntPtr) sizeof(int), M);
        Cl.SetKernelArg(kernel, 4, (IntPtr) sizeof(int), K);
        Cl.SetKernelArg(kernel, 5, (IntPtr) sizeof(int), N);

        IntPtr[] globalWorkSize = { (IntPtr)N, (IntPtr)M };
        Cl.EnqueueNDRangeKernel(_service.CommandQueue.Value, kernel, 2, null, globalWorkSize, null, 0, null, out _);
    }
    
    public void Add(IMathTensor a, IMathTensor b, IMathTensor result)
        => ExecuteElementwiseKernel("elementwise_add_forward", a, b, result);

    public void Multiply(IMathTensor a, IMathTensor b, IMathTensor result)
        => ExecuteElementwiseKernel("elementwise_multiply", a, b, result);
        
    public void Sigmoid(IMathTensor input, IMathTensor result)
        => ExecuteActivationKernel("sigmoid_forward", input, result);

    public void Tanh(IMathTensor input, IMathTensor result)
        => ExecuteActivationKernel("tanh_forward", input, result);

    public void AddBroadcast(IMathTensor matrix, IMathTensor vector, IMathTensor result)
    {
        if (matrix is not GpuTensor gMat || vector is not GpuTensor gVec || result is not GpuTensor gRes)
            throw new ArgumentException("Os tensores devem ser GpuTensors.");
            
        int M = gMat.Shape[0];
        int N = gMat.Shape[1];

        Kernel kernel = _service.Kernels["elementwise_add_broadcast_forward"];
        Cl.SetKernelArg(kernel, 0, (IntPtr)IntPtr.Size, gMat.Buffer);
        Cl.SetKernelArg(kernel, 1, (IntPtr)IntPtr.Size, gVec.Buffer);
        Cl.SetKernelArg(kernel, 2, (IntPtr)IntPtr.Size, gRes.Buffer);
        Cl.SetKernelArg(kernel, 3, (IntPtr)sizeof(int), M);
        Cl.SetKernelArg(kernel, 4, (IntPtr)sizeof(int), N);

        IntPtr[] globalWorkSize = { (IntPtr)N, (IntPtr)M };
        Cl.EnqueueNDRangeKernel(_service.CommandQueue.Value, kernel, 2, null, globalWorkSize, null, 0, null, out _);
    }

    private void ExecuteElementwiseKernel(string kernelName, IMathTensor a, IMathTensor b, IMathTensor result)
    {
        if (a is not GpuTensor gA || b is not GpuTensor gB || result is not GpuTensor gC)
            throw new ArgumentException("Os tensores devem ser GpuTensors.");

        Kernel kernel = _service.Kernels[kernelName];
        Cl.SetKernelArg(kernel, 0, (IntPtr)IntPtr.Size, gA.Buffer);
        Cl.SetKernelArg(kernel, 1, (IntPtr)IntPtr.Size, gB.Buffer);
        Cl.SetKernelArg(kernel, 2, (IntPtr)IntPtr.Size, gC.Buffer);
        Cl.SetKernelArg(kernel, 3, (IntPtr)sizeof(int), gA.TotalSize);
        
        IntPtr[] globalWorkSize = { (IntPtr)gA.TotalSize };
        Cl.EnqueueNDRangeKernel(_service.CommandQueue.Value, kernel, 1, null, globalWorkSize, null, 0, null, out _);
    }
    
    private void ExecuteActivationKernel(string kernelName, IMathTensor input, IMathTensor result)
    {
        if (input is not GpuTensor gInput || result is not GpuTensor gResult)
            throw new ArgumentException("Os tensores devem ser GpuTensors.");
            
        Kernel kernel = _service.Kernels[kernelName];
        Cl.SetKernelArg(kernel, 0, (IntPtr)IntPtr.Size, gInput.Buffer);
        Cl.SetKernelArg(kernel, 1, (IntPtr)IntPtr.Size, gResult.Buffer);
        Cl.SetKernelArg(kernel, 2, (IntPtr)sizeof(int), gInput.TotalSize);

        IntPtr[] globalWorkSize = { (IntPtr)gInput.TotalSize };
        Cl.EnqueueNDRangeKernel(_service.CommandQueue.Value, kernel, 1, null, globalWorkSize, null, 0, null, out _);
    }
    
    public void Dispose() { /* OpenCLService é singleton, não fazemos nada aqui */ }
}