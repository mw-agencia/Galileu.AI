using Galileu.Node.Core;

namespace Galileu.Node.Interfaces;

public interface IMathTensor : IDisposable { }

public interface IMathEngine : IDisposable
{
    bool IsGpu { get; }

    IMathTensor CreateTensor(double[] hostData, int[] shape);
    Tensor ReadTensor(IMathTensor tensor); // Retorna o Tensor da CPU

    void MatrixMultiply(IMathTensor a, IMathTensor b, IMathTensor result);
    void Add(IMathTensor a, IMathTensor b, IMathTensor result);
    void AddBroadcast(IMathTensor matrix, IMathTensor vector, IMathTensor result);
    void Multiply(IMathTensor a, IMathTensor b, IMathTensor result);
    void Sigmoid(IMathTensor input, IMathTensor result);
    void Tanh(IMathTensor input, IMathTensor result);
}