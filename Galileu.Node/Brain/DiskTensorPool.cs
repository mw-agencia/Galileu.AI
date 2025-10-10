using Galileu.Node.Interfaces;
using System;
using System.Collections.Generic;
using System.IO;
using System.IO.MemoryMappedFiles;

namespace Galileu.Node.Brain;

/// <summary>
/// TensorPool que armazena TODOS os tensores em disco (MemoryMappedFile).
/// RAM usada: APENAS o tensor sendo processado no momento (~1-5MB).
/// </summary>
public class DiskTensorPool : IDisposable
{
    private readonly IMathEngine _mathEngine;
    private readonly string _poolFilePath;
    private readonly MemoryMappedFile _mmf;
    private readonly MemoryMappedViewAccessor _accessor;
    
    // Metadados dos tensores (pequeno, fica em RAM)
    private readonly Dictionary<string, Queue<TensorMetadata>> _availableTensors;
    private readonly Dictionary<int, TensorMetadata> _tensorsInUse; // Por ID
    
    private long _currentOffset = 0;
    private int _nextTensorId = 0;
    private readonly object _lock = new object();
    private bool _disposed = false;

    // Configurações
    private const long POOL_SIZE_GB = 20; // 20GB de espaço em disco
    private const long POOL_SIZE_BYTES = POOL_SIZE_GB * 1024 * 1024 * 1024;

    public DiskTensorPool(IMathEngine mathEngine, string poolDirectory = null)
    {
        _mathEngine = mathEngine;
        
        poolDirectory = Path.Combine(Environment.CurrentDirectory, "Dayson", "TensorPool");
        Directory.CreateDirectory(poolDirectory);
        
        _poolFilePath = Path.Combine(poolDirectory, $"tensor_pool_{Guid.NewGuid()}.bin");
        
        Console.WriteLine($"[DiskTensorPool] Criando pool em disco: {_poolFilePath}");
        Console.WriteLine($"[DiskTensorPool] Tamanho máximo: {POOL_SIZE_GB}GB");
        
        // Cria arquivo memory-mapped
        _mmf = MemoryMappedFile.CreateFromFile(
            _poolFilePath,
            FileMode.Create,
            null,
            POOL_SIZE_BYTES,
            MemoryMappedFileAccess.ReadWrite
        );
        
        _accessor = _mmf.CreateViewAccessor(0, POOL_SIZE_BYTES);
        
        _availableTensors = new Dictionary<string, Queue<TensorMetadata>>();
        _tensorsInUse = new Dictionary<int, TensorMetadata>();
        
        Console.WriteLine("[DiskTensorPool] Pool em disco pronto. RAM usada: ~10MB");
    }

    /// <summary>
    /// Aluga um tensor. Retorna um proxy que carrega/salva do disco automaticamente.
    /// </summary>
    public DiskTensorProxy Rent(int[] shape)
    {
        lock (_lock)
        {
            string shapeKey = GetShapeKey(shape);
            
            // Tenta reutilizar tensor existente
            if (_availableTensors.TryGetValue(shapeKey, out var queue) && queue.Count > 0)
            {
                var metadata = queue.Dequeue();
                metadata.InUse = true;
                _tensorsInUse[metadata.Id] = metadata;
                
                return new DiskTensorProxy(this, metadata, _mathEngine);
            }
            
            // Cria novo tensor no disco
            long sizeBytes = CalculateSizeBytes(shape);
            
            if (_currentOffset + sizeBytes > POOL_SIZE_BYTES)
            {
                // Pool cheio - força remoção de tensores antigos
                CompactPool();
            }
            
            var newMetadata = new TensorMetadata
            {
                Id = _nextTensorId++,
                Offset = _currentOffset,
                Shape = shape,
                SizeBytes = sizeBytes,
                InUse = true
            };
            
            _currentOffset += sizeBytes;
            _tensorsInUse[newMetadata.Id] = newMetadata;
            
            return new DiskTensorProxy(this, newMetadata, _mathEngine);
        }
    }

    /// <summary>
    /// Devolve tensor ao pool.
    /// </summary>
    public void Return(DiskTensorProxy proxy)
    {
        lock (_lock)
        {
            var metadata = proxy.Metadata;
            
            if (!_tensorsInUse.ContainsKey(metadata.Id))
                return; // Já foi devolvido
            
            _tensorsInUse.Remove(metadata.Id);
            metadata.InUse = false;
            
            string shapeKey = GetShapeKey(metadata.Shape);
            
            if (!_availableTensors.ContainsKey(shapeKey))
                _availableTensors[shapeKey] = new Queue<TensorMetadata>();
            
            _availableTensors[shapeKey].Enqueue(metadata);
        }
    }

    /// <summary>
    /// Lê tensor do disco.
    /// </summary>
    internal IMathTensor ReadTensor(TensorMetadata metadata)
    {
        lock (_lock)
        {
            int totalElements = metadata.Shape.Aggregate(1, (a, b) => a * b);
            var data = new double[totalElements];
            
            // Lê do memory-mapped file
            long offset = metadata.Offset;
            for (int i = 0; i < totalElements; i++)
            {
                data[i] = _accessor.ReadDouble(offset);
                offset += sizeof(double);
            }
            
            return _mathEngine.CreateTensor(data, metadata.Shape);
        }
    }

    /// <summary>
    /// Escreve tensor no disco.
    /// </summary>
    internal void WriteTensor(TensorMetadata metadata, IMathTensor tensor)
    {
        lock (_lock)
        {
            var cpuData = tensor.ToCpuTensor().GetData();
            
            long offset = metadata.Offset;
            for (int i = 0; i < cpuData.Length; i++)
            {
                _accessor.Write(offset, cpuData[i]);
                offset += sizeof(double);
            }
        }
    }

    /// <summary>
    /// Compacta pool quando fica cheio (remove tensores não usados há muito tempo).
    /// </summary>
    private void CompactPool()
    {
        Console.WriteLine("[DiskTensorPool] Pool cheio - executando compactação...");
        
        // Remove 50% dos tensores disponíveis
        int totalRemoved = 0;
        foreach (var queue in _availableTensors.Values)
        {
            int toRemove = queue.Count / 2;
            for (int i = 0; i < toRemove && queue.Count > 0; i++)
            {
                queue.Dequeue();
                totalRemoved++;
            }
        }
        
        Console.WriteLine($"[DiskTensorPool] Removidos {totalRemoved} tensors do pool");
        
        // Se ainda assim não houver espaço, aumenta o arquivo
        if (_currentOffset > POOL_SIZE_BYTES * 0.9)
        {
            Console.WriteLine("[DiskTensorPool] AVISO: Pool >90% cheio. Considere aumentar POOL_SIZE_GB");
        }
    }

    private string GetShapeKey(int[] shape) => string.Join("x", shape);
    
    private long CalculateSizeBytes(int[] shape)
    {
        long elements = shape.Aggregate(1L, (a, b) => a * b);
        return elements * sizeof(double);
    }

    public void PrintStats()
    {
        lock (_lock)
        {
            long totalAvailable = _availableTensors.Values.Sum(q => q.Count);
            long totalInUse = _tensorsInUse.Count;
            long diskUsedMB = _currentOffset / (1024 * 1024);
            
            Console.WriteLine("\n[DiskTensorPool] === ESTATÍSTICAS ===");
            Console.WriteLine($"Tensors disponíveis: {totalAvailable}");
            Console.WriteLine($"Tensors em uso: {totalInUse}");
            Console.WriteLine($"Disco usado: {diskUsedMB}MB / {POOL_SIZE_GB * 1024}MB");
            Console.WriteLine($"RAM usada: ~10MB (metadados)");
            Console.WriteLine("====================================\n");
        }
    }

    public void Dispose()
    {
        if (_disposed) return;
        
        _accessor?.Dispose();
        _mmf?.Dispose();
        
        try
        {
            if (File.Exists(_poolFilePath))
                File.Delete(_poolFilePath);
        }
        catch { }
        
        _disposed = true;
    }
}