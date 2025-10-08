using Galileu.Node.Core;
using Galileu.Node.Interfaces;

namespace Galileu.Node.Cpu;

public class CpuTensor : IMathTensor
{
    private double[] _data;
    public int[] Shape { get; }

    // CORREÇÃO 2: Propriedade 'Length' implementada conforme exigido pela interface.
    public long Length { get; }

    // CORREÇÃO 1: Propriedade 'IsGpu' implementada conforme exigido pela interface.
    public bool IsGpu => false;

    public CpuTensor(double[] data, int[] shape)
    {
        _data = data;
        Shape = shape;
        Length = data.Length;
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
    /// Implementa o método UpdateFromCpu exigido pela interface.
    /// </summary>
    public void UpdateFromCpu(double[] data)
    {
        if (data.Length != this.Length)
        {
            throw new ArgumentException("Os dados de entrada devem ter o mesmo tamanho do tensor.");
        }

        // Copia os novos dados para o array interno.
        Array.Copy(data, _data, this.Length);
    }

    /// <summary>
    /// Como o array _data é gerenciado pelo Garbage Collector do .NET,
    /// não há recursos não gerenciados para liberar.
    /// </summary>
    public void Dispose()
    {
        /* Nada a fazer */
    }
}