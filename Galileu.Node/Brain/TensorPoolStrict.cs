using Galileu.Node.Interfaces;
using System.Collections.Generic;

namespace Galileu.Node.Brain;

/// <summary>
/// Pool de tensores com LIMITE RÍGIDO e política FAIL-FAST.
/// Se o limite for excedido, PARA o treinamento (não aloca silenciosamente).
/// </summary>
public class TensorPoolStrict : ITensorPool
{
    private readonly IMathEngine _mathEngine;
    private readonly Dictionary<string, Queue<IMathTensor>> _pools;
    private readonly HashSet<IMathTensor> _inUse;
    private bool _disposed = false;

    // ✅ LIMITES RÍGIDOS (NÃO ULTRAPASSÁVEIS)
    private const int MAX_POOL_SIZE_PER_SHAPE = 32;      // Máximo por shape (reduzido de 64)
    private const long MAX_TOTAL_MEMORY_MB = 2048;        // 2GB ABSOLUTO
    private const int MAX_TENSORS_IN_USE = 500;           // Limite de tensores simultâneos
    
    // Estatísticas
    private int _totalRentCalls = 0;
    private int _totalReturnCalls = 0;
    private int _totalAllocations = 0;
    private int _totalReuses = 0;

    public TensorPoolStrict(IMathEngine mathEngine)
    {
        _mathEngine = mathEngine;
        _pools = new Dictionary<string, Queue<IMathTensor>>();
        _inUse = new HashSet<IMathTensor>();
        
        Console.WriteLine($"[TensorPoolStrict] Inicializado com limites rígidos:");
        Console.WriteLine($"  - Máximo por shape: {MAX_POOL_SIZE_PER_SHAPE}");
        Console.WriteLine($"  - Máximo total: {MAX_TOTAL_MEMORY_MB}MB");
        Console.WriteLine($"  - Máximo em uso: {MAX_TENSORS_IN_USE}");
    }

    /// <summary>
    /// CRÍTICO: Aluga tensor com verificação RÍGIDA de limites.
    /// Se exceder, LANÇA EXCEÇÃO (não aloca silenciosamente).
    /// </summary>
    public IMathTensor Rent(int[] shape)
    {
        _totalRentCalls++;
        
        // ✅ VERIFICAÇÃO 1: Número de tensores em uso
        if (_inUse.Count >= MAX_TENSORS_IN_USE)
        {
            throw new OutOfMemoryException(
                $"[TensorPoolStrict] LIMITE EXCEDIDO: {_inUse.Count} tensores em uso " +
                $"(máximo: {MAX_TENSORS_IN_USE}). Possível vazamento de memória!");
        }
        
        string key = GetKey(shape);
        
        if (!_pools.ContainsKey(key))
            _pools[key] = new Queue<IMathTensor>();
        
        var pool = _pools[key];
        
        IMathTensor tensor;
        if (pool.Count > 0)
        {
            // ✅ REUSO: Pega do pool existente
            tensor = pool.Dequeue();
            _totalReuses++;
        }
        else
        {
            // ✅ VERIFICAÇÃO 2: Memória total ANTES de alocar
            long currentMemoryMB = GetTotalMemoryUsageMB();
            long tensorSizeMB = CalculateTensorSizeMB(shape);
            
            if (currentMemoryMB + tensorSizeMB > MAX_TOTAL_MEMORY_MB)
            {
                // ✅ CRÍTICO: Tenta limpeza de emergência ANTES de falhar
                Console.WriteLine($"[TensorPoolStrict] ALERTA: Tentando alocar {tensorSizeMB}MB " +
                                $"mas pool já tem {currentMemoryMB}MB");
                
                TrimExcessMemory(aggressive: true);
                
                // Recalcula após trim
                currentMemoryMB = GetTotalMemoryUsageMB();
                
                if (currentMemoryMB + tensorSizeMB > MAX_TOTAL_MEMORY_MB)
                {
                    PrintDetailedStats();
                    throw new OutOfMemoryException(
                        $"[TensorPoolStrict] LIMITE DE MEMÓRIA EXCEDIDO: " +
                        $"Pool={currentMemoryMB}MB + Novo={tensorSizeMB}MB > Limite={MAX_TOTAL_MEMORY_MB}MB. " +
                        $"Verifique vazamentos ou reduza batch size!");
                }
            }
            
            // ✅ Aloca novo tensor
            tensor = _mathEngine.CreateTensor(shape);
            _totalAllocations++;
        }
        
        _inUse.Add(tensor);
        return tensor;
    }

    /// <summary>
    /// Devolve tensor ao pool com verificação de limites.
    /// </summary>
    public void Return(IMathTensor tensor)
    {
        if (tensor == null || !_inUse.Contains(tensor))
            return;
        
        _totalReturnCalls++;
        _inUse.Remove(tensor);
        
        string key = GetKey(tensor.Shape);
        
        if (!_pools.ContainsKey(key))
            _pools[key] = new Queue<IMathTensor>();
        
        var pool = _pools[key];
        
        // ✅ VERIFICAÇÃO: Se pool deste shape está cheio, LIBERA
        if (pool.Count >= MAX_POOL_SIZE_PER_SHAPE)
        {
            tensor.Dispose();
            return;
        }
        
        pool.Enqueue(tensor);
    }

    /// <summary>
    /// Limpeza agressiva de memória.
    /// </summary>
    private void TrimExcessMemory(bool aggressive = false)
    {
        long before = GetTotalMemoryUsageMB();
        
        if (aggressive)
        {
            Console.WriteLine($"[TensorPoolStrict] TRIM AGRESSIVO: Liberando TODOS os pools...");
            
            // Libera TUDO que não está em uso
            foreach (var pool in _pools.Values)
            {
                while (pool.Count > 0)
                {
                    var tensor = pool.Dequeue();
                    tensor.Dispose();
                }
            }
        }
        else
        {
            // Trim normal: mantém 25% de cada pool
            foreach (var (shape, pool) in _pools)
            {
                int keepCount = Math.Max(4, pool.Count / 4);
                
                while (pool.Count > keepCount)
                {
                    var tensor = pool.Dequeue();
                    tensor.Dispose();
                }
            }
        }
        
        // Força GC
        GC.Collect(2, GCCollectionMode.Forced, true, true);
        GC.WaitForPendingFinalizers();
        
        long after = GetTotalMemoryUsageMB();
        Console.WriteLine($"[TensorPoolStrict] Trim concluído: {before}MB → {after}MB (liberado: {before - after}MB)");
    }

    /// <summary>
    /// Limpa TODOS os tensores não utilizados (entre épocas).
    /// </summary>
    public void Trim()
    {
        Console.WriteLine($"[TensorPoolStrict] Trim manual: Liberando todos os pools...");
        
        foreach (var pool in _pools.Values)
        {
            while (pool.Count > 0)
            {
                var tensor = pool.Dequeue();
                tensor.Dispose();
            }
        }
        
        GC.Collect();
        GC.WaitForPendingFinalizers();
        
        Console.WriteLine($"[TensorPoolStrict] Trim concluído. Tensores em uso: {_inUse.Count}");
    }

    /// <summary>
    /// Calcula uso REAL de memória (CORRIGIDO).
    /// </summary>
    private long GetTotalMemoryUsageMB()
    {
        long totalBytes = 0;
        
        // Pools
        foreach (var (shapeKey, pool) in _pools)
        {
            long shapeBytes = GetShapeMemoryBytes(shapeKey);
            totalBytes += shapeBytes * pool.Count;
        }
        
        // Tensores em uso
        foreach (var tensor in _inUse)
        {
            totalBytes += tensor.Length * sizeof(double); // ✅ CORRIGIDO: sizeof(double)!
        }
        
        return totalBytes / (1024 * 1024);
    }

    /// <summary>
    /// Calcula tamanho de um tensor específico.
    /// </summary>
    private long CalculateTensorSizeMB(int[] shape)
    {
        long elements = shape.Aggregate(1L, (a, b) => a * b);
        long bytes = elements * sizeof(double); // ✅ CORRIGIDO: sizeof(double)!
        return bytes / (1024 * 1024);
    }

    private long GetShapeMemoryBytes(string shapeKey)
    {
        var dimensions = shapeKey.Split('x').Select(int.Parse).ToArray();
        long elements = dimensions.Aggregate(1L, (a, b) => a * b);
        return elements * sizeof(double); // ✅ CORRIGIDO: sizeof(double)!
    }

    /// <summary>
    /// Imprime estatísticas detalhadas (para debug).
    /// </summary>
    public void PrintDetailedStats()
    {
        Console.WriteLine("\n[TensorPoolStrict] ========== ESTATÍSTICAS DETALHADAS ==========");
        Console.WriteLine($"Chamadas Rent():   {_totalRentCalls:N0}");
        Console.WriteLine($"Chamadas Return(): {_totalReturnCalls:N0}");
        Console.WriteLine($"Alocações novas:   {_totalAllocations:N0}");
        Console.WriteLine($"Reusos:            {_totalReuses:N0}");
        Console.WriteLine($"Taxa de reuso:     {(_totalReuses * 100.0 / Math.Max(_totalRentCalls, 1)):F1}%");
        Console.WriteLine($"Tensores em uso:   {_inUse.Count}");
        Console.WriteLine($"Pools ativos:      {_pools.Count}");
        
        long totalPooledMB = 0;
        foreach (var (shape, pool) in _pools.OrderByDescending(kvp => kvp.Value.Count))
        {
            if (pool.Count > 0)
            {
                long shapeMB = GetShapeMemoryBytes(shape) / (1024 * 1024);
                long poolMB = shapeMB * pool.Count;
                totalPooledMB += poolMB;
                Console.WriteLine($"  Shape {shape}: {pool.Count} tensors × {shapeMB}MB = {poolMB}MB");
            }
        }
        
        long inUseMB = _inUse.Sum(t => t.Length * sizeof(double)) / (1024 * 1024);
        
        Console.WriteLine($"\nTotal pooled: {totalPooledMB}MB");
        Console.WriteLine($"Total em uso: {inUseMB}MB");
        Console.WriteLine($"TOTAL POOL:   {totalPooledMB + inUseMB}MB / {MAX_TOTAL_MEMORY_MB}MB");
        Console.WriteLine("=============================================================\n");
    }

    private string GetKey(int[] shape)
    {
        return string.Join("x", shape);
    }

    public void Dispose()
    {
        if (_disposed) return;
        
        // Libera tensores em uso (AVISO: pode indicar vazamento!)
        if (_inUse.Count > 0)
        {
            Console.WriteLine($"[TensorPoolStrict] AVISO: {_inUse.Count} tensores ainda em uso no Dispose!");
        }
        
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
        
        Console.WriteLine($"[TensorPoolStrict] Finalizado. Taxa de reuso: {(_totalReuses * 100.0 / Math.Max(_totalRentCalls, 1)):F1}%");
        
        _disposed = true;
    }
}