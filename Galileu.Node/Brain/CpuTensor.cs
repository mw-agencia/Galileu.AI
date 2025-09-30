using Galileu.Node.Core;
using Galileu.Node.Interfaces;

namespace Galileu.Node.Brain;

public class CpuTensor : IMathTensor
{
    private readonly double[] _data;
    public int[] Shape { get; }
    public int TotalSize { get; }

    public CpuTensor(double[] data, int[] shape)
    {
        _data = data;
        Shape = shape;
        TotalSize = data.Length;
    }

    /// <summary>
    /// Retorna uma referência direta ao array de dados interno.
    /// </summary>
    public double[] GetData() => _data;

    /// <summary>
    /// Cria e retorna um objeto Tensor da CPU, copiando os dados.
    /// </summary>
    public Tensor ToCpuTensor() => new Tensor((double[])_data.Clone(), Shape);

    /// <summary>
    /// Como o array _data é gerenciado pelo Garbage Collector do .NET,
    /// não há recursos não gerenciados para liberar.
    /// </summary>
    public void Dispose() { /* Nada a fazer */ }
}