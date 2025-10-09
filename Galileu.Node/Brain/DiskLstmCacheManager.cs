using Galileu.Node.Core;
using Galileu.Node.Interfaces;
using System;
using System.Collections.Generic;
using System.IO;
using System.IO.MemoryMappedFiles;
using System.Linq;

namespace Galileu.Node.Brain;

/// <summary>
/// Gerencia um cache de tensores LSTM estritamente baseado em disco ("Out-of-Core").
/// - Serializa todos os tensores de ativação para um MemoryMappedFile durante o forward pass.
/// - Não mantém nenhum tensor em cache de RAM, garantindo um consumo de memória mínimo e constante.
/// - Deserializa tensores sob demanda do disco durante o backward pass.
/// </summary>
public class DiskLstmCacheManager : IDisposable
{
    // Tamanho do arquivo de cache em disco. 10GB é um valor seguro para sequências longas.
    private const long DISK_CACHE_SIZE = 20L * 1024 * 1024 * 1024;

    private readonly IMathEngine _mathEngine;
    private readonly MemoryMappedFile _mmf;
    private readonly MemoryMappedViewAccessor _accessor;
    private long _currentDiskPosition = 0;
    private readonly string _tempFilePath;

    // A única estrutura de dados: mapeia timestep e nome do tensor para sua localização no disco.
    private readonly List<Dictionary<string, long>> _diskOffsets;
    private readonly Dictionary<string, int[]> _tensorShapes;
    private bool _disposed = false;

    public DiskLstmCacheManager(IMathEngine mathEngine, int embeddingSize, int hiddenSize, int totalTimesteps)
    {
        _mathEngine = mathEngine;
        _tempFilePath = Path.Combine(Environment.CurrentDirectory, "Dayson", $"lstm_disk_cache_{Guid.NewGuid()}.bin");
        _mmf = MemoryMappedFile.CreateFromFile(_tempFilePath, FileMode.Create, null, DISK_CACHE_SIZE, MemoryMappedFileAccess.ReadWrite);
        _accessor = _mmf.CreateViewAccessor();

        // Pré-aloca a lista para evitar realocações, inicializando com nulos.
        _diskOffsets = Enumerable.Repeat<Dictionary<string, long>>(null!, totalTimesteps).ToList();

        _tensorShapes = new Dictionary<string, int[]>
        {
            { "Input", new[] { 1, embeddingSize } }, { "HiddenPrev", new[] { 1, hiddenSize } },
            { "CellPrev", new[] { 1, hiddenSize } }, { "ForgetGate", new[] { 1, hiddenSize } },
            { "InputGate", new[] { 1, hiddenSize } }, { "CellCandidate", new[] { 1, hiddenSize } },
            { "OutputGate", new[] { 1, hiddenSize } }, { "CellNext", new[] { 1, hiddenSize } },
            { "TanhCellNext", new[] { 1, hiddenSize } }, { "HiddenNext", new[] { 1, hiddenSize } }
        };
    }

    /// <summary>
    /// Serializa todos os tensores de um LstmStepCache diretamente para o disco.
    /// </summary>
    public void CacheStep(LstmStepCache gpuStepCache, int timeStep)
    {
        var currentStepOffsets = new Dictionary<string, long>();

        // Itera sobre cada tensor, escreve no disco e armazena sua localização.
        currentStepOffsets["Input"] = WriteTensorToDisk(gpuStepCache.Input!);
        currentStepOffsets["HiddenPrev"] = WriteTensorToDisk(gpuStepCache.HiddenPrev!);
        currentStepOffsets["CellPrev"] = WriteTensorToDisk(gpuStepCache.CellPrev!);
        currentStepOffsets["ForgetGate"] = WriteTensorToDisk(gpuStepCache.ForgetGate!);
        currentStepOffsets["InputGate"] = WriteTensorToDisk(gpuStepCache.InputGate!);
        currentStepOffsets["CellCandidate"] = WriteTensorToDisk(gpuStepCache.CellCandidate!);
        currentStepOffsets["OutputGate"] = WriteTensorToDisk(gpuStepCache.OutputGate!);
        currentStepOffsets["CellNext"] = WriteTensorToDisk(gpuStepCache.CellNext!);
        currentStepOffsets["TanhCellNext"] = WriteTensorToDisk(gpuStepCache.TanhCellNext!);
        currentStepOffsets["HiddenNext"] = WriteTensorToDisk(gpuStepCache.HiddenNext!);

        _diskOffsets[timeStep] = currentStepOffsets;
    }

    /// <summary>
    /// Recupera um tensor lendo-o diretamente do cache em disco.
    /// </summary>
    public IMathTensor RetrieveTensor(int timeStep, string tensorName)
    {
        long offset = _diskOffsets[timeStep][tensorName];
        var shape = _tensorShapes[tensorName];
        
        int length = _accessor.ReadInt32(offset);
        var data = new double[length];
        _accessor.ReadArray(offset + sizeof(int), data, 0, length);

        return _mathEngine.CreateTensor(data, shape);
    }

    /// <summary>
    /// Serializa um IMathTensor para o MemoryMappedFile.
    /// </summary>
    private long WriteTensorToDisk(IMathTensor tensor)
    {
        // Converte para CPU para obter os dados brutos.
        var data = tensor.ToCpuTensor().GetData();
        int length = data.Length;
        int byteLength = length * sizeof(double);
        long startOffset = _currentDiskPosition;

        // Verifica se há espaço suficiente no arquivo mapeado.
        if (startOffset + sizeof(int) + byteLength > DISK_CACHE_SIZE)
        {
            throw new OutOfMemoryException("O cache em disco excedeu o tamanho alocado de 10GB.");
        }

        _accessor.Write(startOffset, length);
        _accessor.WriteArray(startOffset + sizeof(int), data, 0, length);
        
        _currentDiskPosition += sizeof(int) + byteLength;
        return startOffset;
    }

    /// <summary>
    /// Limpa o estado do cache para uma nova sequência.
    /// </summary>
    public void Reset()
    {
        _currentDiskPosition = 0;
    }

    /// <summary>
    /// Libera todos os recursos.
    /// </summary>
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        
        _accessor?.Dispose();
        _mmf?.Dispose();

        try
        {
            if (File.Exists(_tempFilePath))
            {
                File.Delete(_tempFilePath);
            }
        }
        catch (IOException) { /* Ignora erros se o arquivo estiver bloqueado */ }
        
        GC.SuppressFinalize(this);
    }
}