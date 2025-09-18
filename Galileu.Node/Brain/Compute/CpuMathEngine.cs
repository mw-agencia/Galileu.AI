// Local: Brain/Compute/CpuMathEngine.cs
using Galileu.Node.Core;
using Galileu.Node.Interfaces;

namespace Galileu.Node.Brain.Compute;

public class CpuMathEngine : IMathEngine
{
    private class CpuTensor : IMathTensor
    {
        public double[] Data { get; }
        public int[] Shape { get; }
        public int TotalSize => Data.Length;
        public CpuTensor(double[] data, int[] shape) { Data = data; Shape = shape; }
        public void Dispose() { } // Nada para liberar na CPU
    }

    public bool IsGpu => false;

    public IMathTensor CreateTensor(double[] hostData, int[] shape)
    {
        return new CpuTensor((double[])hostData.Clone(), shape);
    }

    public Tensor ReadTensor(IMathTensor tensor)
    {
        var cpuTensor = (CpuTensor)tensor;
        return new Tensor((double[])cpuTensor.Data.Clone(), cpuTensor.Shape);
    }

    public void MatrixMultiply(IMathTensor a, IMathTensor b, IMathTensor result)
    {
        var A = ((CpuTensor)a).Data; var B = ((CpuTensor)b).Data; var C = ((CpuTensor)result).Data;
        int M = a.Shape[0]; int K = a.Shape[1]; int N = b.Shape[1];
        for (int r = 0; r < M; r++)
            for (int c = 0; c < N; c++)
            {
                double sum = 0;
                for (int k = 0; k < K; k++) sum += A[r * K + k] * B[k * N + c];
                C[r * N + c] = sum;
            }
    }
    
    // ... Implemente os outros métodos (Add, Multiply, etc.) com a lógica C# ...
    public void Add(IMathTensor a, IMathTensor b, IMathTensor result)
    {
        var A = ((CpuTensor)a).Data; var B = ((CpuTensor)b).Data; var C = ((CpuTensor)result).Data;
        for (int i = 0; i < A.Length; i++) C[i] = A[i] + B[i];
    }
    public void AddBroadcast(IMathTensor matrix, IMathTensor vector, IMathTensor result)
    {
        var M = ((CpuTensor)matrix).Data; var V = ((CpuTensor)vector).Data; var R = ((CpuTensor)result).Data;
        int rows = matrix.Shape[0]; int cols = matrix.Shape[1];
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < cols; c++) R[r * cols + c] = M[r * cols + c] + V[c];
    }
    public void Multiply(IMathTensor a, IMathTensor b, IMathTensor result)
    {
        var A = ((CpuTensor)a).Data; var B = ((CpuTensor)b).Data; var C = ((CpuTensor)result).Data;
        for (int i = 0; i < A.Length; i++) C[i] = A[i] * B[i];
    }
    public void Sigmoid(IMathTensor input, IMathTensor result)
    {
        var I = ((CpuTensor)input).Data; var R = ((CpuTensor)result).Data;
        for (int i = 0; i < I.Length; i++) R[i] = 1.0 / (1.0 + Math.Exp(-I[i]));
    }
    public void Tanh(IMathTensor input, IMathTensor result)
    {
        var I = ((CpuTensor)input).Data; var R = ((CpuTensor)result).Data;
        for (int i = 0; i < I.Length; i++) R[i] = Math.Tanh(I[i]);
    }

    public void Dispose() { }
}