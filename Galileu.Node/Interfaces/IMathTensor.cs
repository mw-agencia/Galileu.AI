using Galileu.Node.Core;

namespace Galileu.Node.Interfaces;

public interface IMathTensor : IDisposable
{
    int[] Shape { get; }
    int TotalSize { get; }
    Tensor ToCpuTensor(); // Método para ler os dados de volta para a CPU
}

public interface IMathEngine : IDisposable
{
    bool IsGpu { get; }

    IMathTensor CreateTensor(int[] shape);
    IMathTensor CreateTensor(double[] hostData, int[] shape);
    
    void MatrixMultiply(IMathTensor a, IMathTensor b, IMathTensor result);
    void Add(IMathTensor a, IMathTensor b, IMathTensor result);
    void AddBroadcast(IMathTensor matrix, IMathTensor vector, IMathTensor result);
    void Multiply(IMathTensor a, IMathTensor b, IMathTensor result);
    void Sigmoid(IMathTensor input, IMathTensor result);
    void Tanh(IMathTensor input, IMathTensor result);
}