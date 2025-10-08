using System;
using System.Diagnostics;
using System.IO;
using System.IO.MemoryMappedFiles;
using System.Runtime.InteropServices;
using Galileu.Node.Interfaces;

namespace Galileu.Node.Brain;

/// <summary>
/// Gerencia caches de timesteps LSTM em memória virtual (disco) para reduzir consumo de RAM.
/// Usa MemoryMappedFile para acesso eficiente e serialização binária customizada.
/// </summary>
public class VirtualCacheManager : IDisposable
{
    private readonly string _cacheFilePath;
    private MemoryMappedFile? _memoryMappedFile;
    private readonly long _maxCacheSize;
    private long _currentOffset;
    private readonly Dictionary<int, CacheMetadata> _cacheIndex;
    private readonly IMathEngine _mathEngine;
    private bool _disposed = false;

    // Mantém os últimos N caches em memória para acesso rápido
    private readonly int _ramCacheSize;
    private readonly Dictionary<int, LstmStepCache> _ramCache;
    private readonly Queue<int> _ramCacheQueue;

    private struct CacheMetadata
    {
        public long Offset;
        public int DataSize;
        public int[] InputShape;
        public int[] HiddenShape;
    }

    public VirtualCacheManager(IMathEngine mathEngine, int ramCacheSize = 8, long maxCacheSizeBytes = 10L * 1024 * 1024 * 1024) // 10GB padrão
    {
        _mathEngine = mathEngine;
        _ramCacheSize = ramCacheSize;
        _maxCacheSize = maxCacheSizeBytes;
        _currentOffset = 0;
        _cacheIndex = new Dictionary<int, CacheMetadata>();
        _ramCache = new Dictionary<int, LstmStepCache>();
        _ramCacheQueue = new Queue<int>();

        // Cria arquivo temporário para cache
        string tempDir = Path.GetTempPath();
        string fileName = $"lstm_cache_{DateTime.Now:yyyyMMdd_HHmmss}_{Process.GetCurrentProcess().Id}.tmp";
        _cacheFilePath = Path.Combine(tempDir, fileName);
        
        Console.WriteLine($"[VirtualCache] Inicializando cache: {_cacheFilePath}");
        
        try
        {
            // Garante que o diretório existe
            Directory.CreateDirectory(Path.GetDirectoryName(_cacheFilePath)!);
            
            // Cria arquivo físico primeiro para garantir permissões
            using (var fs = File.Create(_cacheFilePath))
            {
                fs.SetLength(_maxCacheSize);
            }
            
            // Agora cria o MemoryMappedFile sobre o arquivo existente
            _memoryMappedFile = MemoryMappedFile.CreateFromFile(
                _cacheFilePath,
                FileMode.Open,
                null,
                _maxCacheSize,
                MemoryMappedFileAccess.ReadWrite);
            
            Console.WriteLine($"[VirtualCache] Cache virtual criado com sucesso (Max: {_maxCacheSize / (1024*1024)}MB)");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[VirtualCache] ERRO CRÍTICO ao criar cache virtual: {ex.GetType().Name}: {ex.Message}");
            Console.WriteLine($"[VirtualCache] Stack trace: {ex.StackTrace}");
            throw new InvalidOperationException($"Falha ao inicializar cache virtual em {_cacheFilePath}", ex);
        }
    }

    /// <summary>
    /// Armazena um LstmStepCache no disco e retorna seu índice.
    /// </summary>
    public int StoreCache(int timestep, LstmStepCache cache)
    {
        if (_disposed) throw new ObjectDisposedException(nameof(VirtualCacheManager));

        try
        {
            // Calcula tamanho necessário
            int tensorCount = 10; // Número de tensores no LstmStepCache
            long tensorSize = cache.Input!.Length * sizeof(float);
            int totalSize = (int)(tensorSize * tensorCount + 1024); // +1KB para metadados

            if (_currentOffset + totalSize > _maxCacheSize)
            {
                throw new OutOfMemoryException($"Cache virtual excedeu limite de {_maxCacheSize / (1024*1024)}MB");
            }

            // Debug log a cada 100 timesteps
            if (timestep % 100 == 0)
            {
                Console.WriteLine($"[VirtualCache] Armazenando timestep {timestep} no offset {_currentOffset} ({totalSize} bytes)");
            }

            // Serializa para disco
            using (var accessor = _memoryMappedFile!.CreateViewAccessor(_currentOffset, totalSize, MemoryMappedFileAccess.Write))
            {
                long writePos = 0;

                // Escreve shapes (metadados)
                WriteIntArray(accessor, ref writePos, cache.Input!.Shape);
                WriteIntArray(accessor, ref writePos, cache.HiddenPrev!.Shape);

                // Serializa cada tensor
                WriteTensor(accessor, ref writePos, cache.Input);
                WriteTensor(accessor, ref writePos, cache.HiddenPrev);
                WriteTensor(accessor, ref writePos, cache.CellPrev);
                WriteTensor(accessor, ref writePos, cache.ForgetGate);
                WriteTensor(accessor, ref writePos, cache.InputGate);
                WriteTensor(accessor, ref writePos, cache.CellCandidate);
                WriteTensor(accessor, ref writePos, cache.OutputGate);
                WriteTensor(accessor, ref writePos, cache.CellNext);
                WriteTensor(accessor, ref writePos, cache.TanhCellNext);
                WriteTensor(accessor, ref writePos, cache.HiddenNext);
            }

            // Registra metadados
            var metadata = new CacheMetadata
            {
                Offset = _currentOffset,
                DataSize = totalSize,
                InputShape = cache.Input.Shape,
                HiddenShape = cache.HiddenPrev!.Shape
            };
            _cacheIndex[timestep] = metadata;

            _currentOffset += totalSize;

            // Libera tensores GPU originais para economizar memória
            //cache.Dispose();

            return timestep;
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[VirtualCache] ERRO ao armazenar timestep {timestep}: {ex.GetType().Name}: {ex.Message}");
            throw;
        }
    }

    /// <summary>
    /// Recupera um LstmStepCache do disco ou da RAM cache.
    /// </summary>
    public LstmStepCache LoadCache(int timestep)
    {
        if (_disposed) throw new ObjectDisposedException(nameof(VirtualCacheManager));

        // Verifica RAM cache primeiro
        if (_ramCache.TryGetValue(timestep, out var cachedStep))
        {
            return cachedStep;
        }

        // Carrega do disco
        if (!_cacheIndex.TryGetValue(timestep, out var metadata))
        {
            throw new KeyNotFoundException($"Cache para timestep {timestep} não encontrado.");
        }

        var cache = new LstmStepCache();

        using (var accessor = _memoryMappedFile!.CreateViewAccessor(metadata.Offset, metadata.DataSize, MemoryMappedFileAccess.Read))
        {
            long readPos = 0;

            // Lê shapes
            int[] inputShape = ReadIntArray(accessor, ref readPos);
            int[] hiddenShape = ReadIntArray(accessor, ref readPos);

            // Deserializa tensores
            cache.Input = ReadTensor(accessor, ref readPos, inputShape);
            cache.HiddenPrev = ReadTensor(accessor, ref readPos, hiddenShape);
            cache.CellPrev = ReadTensor(accessor, ref readPos, hiddenShape);
            cache.ForgetGate = ReadTensor(accessor, ref readPos, hiddenShape);
            cache.InputGate = ReadTensor(accessor, ref readPos, hiddenShape);
            cache.CellCandidate = ReadTensor(accessor, ref readPos, hiddenShape);
            cache.OutputGate = ReadTensor(accessor, ref readPos, hiddenShape);
            cache.CellNext = ReadTensor(accessor, ref readPos, hiddenShape);
            cache.TanhCellNext = ReadTensor(accessor, ref readPos, hiddenShape);
            cache.HiddenNext = ReadTensor(accessor, ref readPos, hiddenShape);
        }

        // Adiciona à RAM cache (LRU)
        AddToRamCache(timestep, cache);

        return cache;
    }

    /// <summary>
    /// Limpa todos os caches e reinicia.
    /// </summary>
    public void Clear()
    {
        foreach (var cache in _ramCache.Values)
        {
            //cache.Dispose();
        }
        _ramCache.Clear();
        _ramCacheQueue.Clear();
        _cacheIndex.Clear();
        _currentOffset = 0;
    }

    private void AddToRamCache(int timestep, LstmStepCache cache)
    {
        if (_ramCache.Count >= _ramCacheSize)
        {
            // Remove cache mais antigo (LRU)
            int oldestTimestep = _ramCacheQueue.Dequeue();
            if (_ramCache.TryGetValue(oldestTimestep, out var oldCache))
            {
                //oldCache.Dispose();
                _ramCache.Remove(oldestTimestep);
            }
        }

        _ramCache[timestep] = cache;
        _ramCacheQueue.Enqueue(timestep);
    }

    // --- Serialização Binária ---

    private void WriteTensor(MemoryMappedViewAccessor accessor, ref long position, IMathTensor? tensor)
    {
        if (tensor == null)
        {
            accessor.Write(position, 0L); // Marca como null
            position += sizeof(long);
            return;
        }

        long length = tensor.Length;
        accessor.Write(position, length);
        position += sizeof(long);

        var cpuData = tensor.ToCpuTensor().GetData();
        for (long i = 0; i < length; i++)
        {
            accessor.Write(position, (float)cpuData[i]);
            position += sizeof(float);
        }
    }

    private IMathTensor? ReadTensor(MemoryMappedViewAccessor accessor, ref long position, int[] shape)
    {
        long length = accessor.ReadInt64(position);
        position += sizeof(long);

        if (length == 0) return null;

        double[] data = new double[length];
        for (long i = 0; i < length; i++)
        {
            data[i] = accessor.ReadSingle(position);
            position += sizeof(float);
        }

        return _mathEngine.CreateTensor(data, shape);
    }

    private void WriteIntArray(MemoryMappedViewAccessor accessor, ref long position, int[] array)
    {
        accessor.Write(position, array.Length);
        position += sizeof(int);

        for (int i = 0; i < array.Length; i++)
        {
            accessor.Write(position, array[i]);
            position += sizeof(int);
        }
    }

    private int[] ReadIntArray(MemoryMappedViewAccessor accessor, ref long position)
    {
        int length = accessor.ReadInt32(position);
        position += sizeof(int);

        int[] array = new int[length];
        for (int i = 0; i < length; i++)
        {
            array[i] = accessor.ReadInt32(position);
            position += sizeof(int);
        }
        return array;
    }

    public void Dispose()
    {
        if (_disposed) return;

        Clear();
        _memoryMappedFile?.Dispose();

        // Remove arquivo temporário
        try
        {
            if (File.Exists(_cacheFilePath))
            {
                File.Delete(_cacheFilePath);
                Console.WriteLine($"[VirtualCache] Cache virtual removido: {_cacheFilePath}");
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[VirtualCache] Aviso: Não foi possível remover arquivo temporário: {ex.Message}");
        }

        _disposed = true;
    }
}