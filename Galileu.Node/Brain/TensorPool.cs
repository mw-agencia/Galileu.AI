using Galileu.Node.Interfaces;
using System.Collections.Generic;

namespace Galileu.Node.Brain;

/// <summary>
/// Pool de tensores reutilizáveis para reduzir alocações durante forward/backward pass.
/// Crítico para evitar fragmentação de memória GPU.
/// </summary>
public class TensorPool : IDisposable
{
    private readonly IMathEngine _mathEngine;
    private readonly Dictionary<string, Queue<IMathTensor>> _pools;
    private readonly HashSet<IMathTensor> _inUse;
    private bool _disposed = false;

    public TensorPool(IMathEngine mathEngine)
    {
        _mathEngine = mathEngine;
        _pools = new Dictionary<string, Queue<IMathTensor>>();
        _inUse = new HashSet<IMathTensor>();
    }

    /// <summary>
    /// Obtém ou cria um tensor com as dimensões especificadas.
    /// </summary>
    public IMathTensor Rent(int[] shape)
    {
        string key = GetKey(shape);
        
        if (!_pools.ContainsKey(key))
            _pools[key] = new Queue<IMathTensor>();
        
        var pool = _pools[key];
        
        IMathTensor tensor;
        if (pool.Count > 0)
        {
            tensor = pool.Dequeue();
        }
        else
        {
            tensor = _mathEngine.CreateTensor(shape);
        }
        
        _inUse.Add(tensor);
        return tensor;
    }

    /// <summary>
    /// Devolve o tensor ao pool para reutilização.
    /// </summary>
    public void Return(IMathTensor tensor)
    {
        if (tensor == null || !_inUse.Contains(tensor))
            return;
        
        _inUse.Remove(tensor);
        string key = GetKey(tensor.Shape);
        
        if (!_pools.ContainsKey(key))
            _pools[key] = new Queue<IMathTensor>();
        
        _pools[key].Enqueue(tensor);
    }

    /// <summary>
    /// Libera todos os tensores que não estão em uso.
    /// Chame isso entre épocas para limpar memória.
    /// </summary>
    public void Trim()
    {
        foreach (var pool in _pools.Values)
        {
            while (pool.Count > 0)
            {
                var tensor = pool.Dequeue();
                tensor.Dispose();
            }
        }
    }

    private string GetKey(int[] shape)
    {
        return string.Join("x", shape);
    }

    public void Dispose()
    {
        if (_disposed) return;
        
        // Libera tensores em uso
        foreach (var tensor in _inUse)
            tensor.Dispose();
        
        // Libera pools
        foreach (var pool in _pools.Values)
        {
            while (pool.Count > 0)
                pool.Dequeue().Dispose();
        }
        
        _inUse.Clear();
        _pools.Clear();
        _disposed = true;
    }
}