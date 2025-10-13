using Galileu.Node.Interfaces;
using System;

namespace Galileu.Node.Brain;

/// <summary>
/// Interface comum para todas as implementações de TensorPool.
/// Permite polimorfismo entre TensorPool e TensorPoolStrict.
/// </summary>
public interface ITensorPool : IDisposable
{
    /// <summary>
    /// Aluga um tensor com as dimensões especificadas.
    /// Se não houver tensor disponível no pool, cria um novo.
    /// </summary>
    /// <param name="shape">Dimensões do tensor (ex: [1, 256])</param>
    /// <returns>Tensor alocado (deve ser devolvido com Return)</returns>
    IMathTensor Rent(int[] shape);

    /// <summary>
    /// Devolve um tensor ao pool para reutilização futura.
    /// O tensor NÃO é liberado (Dispose), apenas devolvido ao pool.
    /// </summary>
    /// <param name="tensor">Tensor a ser devolvido</param>
    void Return(IMathTensor tensor);

    /// <summary>
    /// Libera todos os tensores que NÃO estão em uso (no pool mas não alugados).
    /// Útil para liberar memória entre épocas de treinamento.
    /// </summary>
    void Trim();

    /// <summary>
    /// Imprime estatísticas detalhadas sobre o uso do pool.
    /// Inclui: taxa de reuso, memória alocada, tensores em uso, etc.
    /// </summary>
    void PrintDetailedStats();
}