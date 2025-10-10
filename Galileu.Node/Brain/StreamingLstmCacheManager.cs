using Galileu.Node.Core;
using Galileu.Node.Interfaces;
using System;
using System.Collections.Generic;
using System.IO;

namespace Galileu.Node.Brain;

/// <summary>
/// Cache LSTM que mantém ZERO dados em RAM.
/// Tudo é lido/escrito diretamente do disco via FileStream.
/// Usa indexação eficiente para acesso O(1).
/// </summary>
public class StreamingLstmCacheManager : IDisposable
{
    private readonly IMathEngine _mathEngine;
    private readonly string _cacheFilePath;
    private readonly FileStream _fileStream;
    private readonly BinaryWriter _writer;
    private readonly BinaryReader _reader;
    
    // Apenas metadados em RAM (pequeno: ~16 bytes por timestep)
    private readonly Dictionary<int, Dictionary<string, TensorLocation>> _index;
    private readonly Dictionary<string, int[]> _tensorShapes;
    
    private bool _disposed = false;
    private readonly object _lock = new object();

    private struct TensorLocation
    {
        public long Offset;      // Posição no arquivo
        public int Length;       // Número de elementos
    }

    public StreamingLstmCacheManager(IMathEngine mathEngine, int embeddingSize, int hiddenSize)
    {
        _mathEngine = mathEngine;
        _index = new Dictionary<int, Dictionary<string, TensorLocation>>();
        
        _tensorShapes = new Dictionary<string, int[]>
        {
            { "Input", new[] { 1, embeddingSize } },
            { "HiddenPrev", new[] { 1, hiddenSize } },
            { "CellPrev", new[] { 1, hiddenSize } },
            { "ForgetGate", new[] { 1, hiddenSize } },
            { "InputGate", new[] { 1, hiddenSize } },
            { "CellCandidate", new[] { 1, hiddenSize } },
            { "OutputGate", new[] { 1, hiddenSize } },
            { "CellNext", new[] { 1, hiddenSize } },
            { "TanhCellNext", new[] { 1, hiddenSize } },
            { "HiddenNext", new[] { 1, hiddenSize } }
        };

        // Cria arquivo temporário
        var cacheDir = Path.Combine(Environment.CurrentDirectory, "Dayson", "StreamingCache");
        Directory.CreateDirectory(cacheDir);
        _cacheFilePath = Path.Combine(cacheDir, $"lstm_streaming_{Guid.NewGuid()}.bin");
        
        // FileStream com buffer mínimo (4KB) para economizar RAM
        _fileStream = new FileStream(
            _cacheFilePath,
            FileMode.Create,
            FileAccess.ReadWrite,
            FileShare.None,
            bufferSize: 4096,  // Buffer mínimo
            FileOptions.RandomAccess | FileOptions.WriteThrough
        );
        
        _writer = new BinaryWriter(_fileStream);
        _reader = new BinaryReader(_fileStream);
        
        //Console.WriteLine($"[StreamingCache] Inicializado: {_cacheFilePath}");
        //Console.WriteLine("[StreamingCache] RAM usada: ~5MB (índice)");
    }

    /// <summary>
    /// Salva um timestep completo no disco.
    /// RAM temporária: apenas 1 timestep (~2.5MB) durante gravação.
    /// </summary>
    public void CacheStep(LstmStepCache gpuStepCache, int timeStep)
    {
        lock (_lock)
        {
            if (!_index.ContainsKey(timeStep))
                _index[timeStep] = new Dictionary<string, TensorLocation>();
            
            var stepIndex = _index[timeStep];
            
            // Grava cada tensor diretamente no disco
            stepIndex["Input"] = WriteTensorToDisk(gpuStepCache.Input!);
            stepIndex["HiddenPrev"] = WriteTensorToDisk(gpuStepCache.HiddenPrev!);
            stepIndex["CellPrev"] = WriteTensorToDisk(gpuStepCache.CellPrev!);
            stepIndex["ForgetGate"] = WriteTensorToDisk(gpuStepCache.ForgetGate!);
            stepIndex["InputGate"] = WriteTensorToDisk(gpuStepCache.InputGate!);
            stepIndex["CellCandidate"] = WriteTensorToDisk(gpuStepCache.CellCandidate!);
            stepIndex["OutputGate"] = WriteTensorToDisk(gpuStepCache.OutputGate!);
            stepIndex["CellNext"] = WriteTensorToDisk(gpuStepCache.CellNext!);
            stepIndex["TanhCellNext"] = WriteTensorToDisk(gpuStepCache.TanhCellNext!);
            stepIndex["HiddenNext"] = WriteTensorToDisk(gpuStepCache.HiddenNext!);
            
            // Força flush para disco (evita buffer em RAM)
            _writer.Flush();
        }
    }

    /// <summary>
    /// Recupera um tensor específico do disco.
    /// RAM temporária: apenas ~100KB durante leitura.
    /// </summary>
    public IMathTensor RetrieveTensor(int timeStep, string tensorName)
    {
        lock (_lock)
        {
            if (!_index.TryGetValue(timeStep, out var stepIndex))
                throw new KeyNotFoundException($"Timestep {timeStep} não encontrado");
            
            if (!stepIndex.TryGetValue(tensorName, out var location))
                throw new KeyNotFoundException($"Tensor {tensorName} não encontrado no timestep {timeStep}");
            
            var shape = _tensorShapes[tensorName];
            return ReadTensorFromDisk(location, shape);
        }
    }

    /// <summary>
    /// Escreve tensor no disco e retorna localização.
    /// </summary>
    private TensorLocation WriteTensorToDisk(IMathTensor tensor)
    {
        long startOffset = _fileStream.Position;
        
        // Converte para CPU e pega dados
        var cpuData = tensor.ToCpuTensor().GetData();
        int length = cpuData.Length;
        
        // Escreve: [length:int][data:double[]]
        _writer.Write(length);
        
        // Escreve dados em chunks para evitar buffer grande
        const int CHUNK_SIZE = 1024;
        for (int i = 0; i < length; i += CHUNK_SIZE)
        {
            int chunkLength = Math.Min(CHUNK_SIZE, length - i);
            for (int j = 0; j < chunkLength; j++)
            {
                _writer.Write(cpuData[i + j]);
            }
        }
        
        return new TensorLocation
        {
            Offset = startOffset,
            Length = length
        };
    }

    /// <summary>
    /// Lê tensor do disco.
    /// </summary>
    private IMathTensor ReadTensorFromDisk(TensorLocation location, int[] shape)
    {
        _fileStream.Seek(location.Offset, SeekOrigin.Begin);
        
        int length = _reader.ReadInt32();
        
        if (length != location.Length)
            throw new InvalidDataException($"Tamanho inconsistente: esperado {location.Length}, lido {length}");
        
        var data = new double[length];
        
        // Lê em chunks
        const int CHUNK_SIZE = 1024;
        for (int i = 0; i < length; i += CHUNK_SIZE)
        {
            int chunkLength = Math.Min(CHUNK_SIZE, length - i);
            for (int j = 0; j < chunkLength; j++)
            {
                data[i + j] = _reader.ReadDouble();
            }
        }
        
        return _mathEngine.CreateTensor(data, shape);
    }

    /// <summary>
    /// Limpa cache completamente (chamado entre batches).
    /// </summary>
    public void Reset()
    {
        lock (_lock)
        {
            _index.Clear();
            _fileStream.SetLength(0);
            _fileStream.Flush();
            
            // Força GC para limpar qualquer resíduo de RAM
            GC.Collect(1, GCCollectionMode.Optimized, false);
        }
    }

    /// <summary>
    /// Imprime estatísticas de uso.
    /// </summary>
    public void PrintStats()
    {
        lock (_lock)
        {
            long fileSizeMB = _fileStream.Length / (1024 * 1024);
            int timestepsCached = _index.Count;
            long indexSizeBytes = timestepsCached * 10 * 16; // ~16 bytes por tensor
            long indexSizeMB = indexSizeBytes / (1024 * 1024);
            
            Console.WriteLine($"[StreamingCache] Timesteps cacheados: {timestepsCached}");
            Console.WriteLine($"[StreamingCache] Tamanho do arquivo: {fileSizeMB}MB");
            Console.WriteLine($"[StreamingCache] RAM usada (índice): ~{indexSizeMB}MB");
        }
    }

    public void Dispose()
    {
        if (_disposed) return;
        
        lock (_lock)
        {
            _writer?.Close();
            _reader?.Close();
            _fileStream?.Close();
            
            try
            {
                if (File.Exists(_cacheFilePath))
                    File.Delete(_cacheFilePath);
            }
            catch { }
        }
        
        _disposed = true;
    }
}