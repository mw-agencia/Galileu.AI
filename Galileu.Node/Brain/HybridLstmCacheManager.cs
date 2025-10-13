using Galileu.Node.Core;
using Galileu.Node.Interfaces;
using System.IO.MemoryMappedFiles;

namespace Galileu.Node.Brain;

/// <summary>
/// Cache híbrido RAM+Disco para timesteps LSTM.
/// OTIMIZADO: Limpeza automática de metadados para treinamentos longos.
/// </summary>
public class HybridLstmCacheManager : IDisposable
{
    private const int RAM_CACHE_CAPACITY = 16;
    private const long DISK_CACHE_SIZE = 20L * 1024 * 1024 * 1024; // 10GB

    private readonly IMathEngine _mathEngine;
    private readonly MemoryMappedFile _mmf;
    private readonly MemoryMappedViewAccessor _accessor;
    private long _currentDiskPosition = 0;
    private readonly string _tempFilePath;

    // Cache de RAM (LRU)
    private readonly Dictionary<int, RamCacheItem> _ramCache;
    private readonly LinkedList<int> _lruTracker;
    
    // === OTIMIZAÇÃO: Metadados com limpeza automática ===
    // Este dicionário armazena os "endereços" dos tensores no disco.
    // Era a principal fonte do vazamento de memória antes da implementação do Reset().
    private readonly Dictionary<int, Dictionary<string, long>> _diskOffsets;
    private int _maxTimestepCached = 0;
    
    private readonly Dictionary<string, int[]> _tensorShapes;
    private bool _disposed = false;

    // Classe interna para armazenar tensores na RAM (já convertidos para formato de CPU)
    private class RamCacheItem
    {
        public Tensor? Input { get; set; }
        public Tensor? HiddenPrev { get; set; }
        public Tensor? CellPrev { get; set; }
        public Tensor? ForgetGate { get; set; }
        public Tensor? InputGate { get; set; }
        public Tensor? CellCandidate { get; set; }
        public Tensor? OutputGate { get; set; }
        public Tensor? CellNext { get; set; }
        public Tensor? TanhCellNext { get; set; }
        public Tensor? HiddenNext { get; set; }
    }

    public HybridLstmCacheManager(IMathEngine mathEngine, int embeddingSize, int hiddenSize, int totalTimesteps)
    {
        _mathEngine = mathEngine;
        _tempFilePath = Path.Combine(Environment.CurrentDirectory, "Dayson", $"lstm_hybrid_cache_{Guid.NewGuid()}.bin");
        _mmf = MemoryMappedFile.CreateFromFile(_tempFilePath, FileMode.Create, null, DISK_CACHE_SIZE, MemoryMappedFileAccess.ReadWrite);
        _accessor = _mmf.CreateViewAccessor();

        _ramCache = new Dictionary<int, RamCacheItem>();
        _diskOffsets = new Dictionary<int, Dictionary<string, long>>();
        _lruTracker = new LinkedList<int>();

        // Pré-calcula as formas dos tensores para reconstrução
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
    }

    public void CacheStep(LstmStepCache gpuStepCache, int timeStep)
    {
        // Se a RAM está cheia, move o item menos usado para o disco
        if (_ramCache.Count >= RAM_CACHE_CAPACITY)
        {
            EvictLruItemToDisk();
        }

        // Converte os tensores da GPU para tensores de CPU para armazenamento
        var ramItem = new RamCacheItem
        {
            Input = gpuStepCache.Input!.ToCpuTensor(),
            HiddenPrev = gpuStepCache.HiddenPrev!.ToCpuTensor(),
            CellPrev = gpuStepCache.CellPrev!.ToCpuTensor(),
            ForgetGate = gpuStepCache.ForgetGate!.ToCpuTensor(),
            InputGate = gpuStepCache.InputGate!.ToCpuTensor(),
            CellCandidate = gpuStepCache.CellCandidate!.ToCpuTensor(),
            OutputGate = gpuStepCache.OutputGate!.ToCpuTensor(),
            CellNext = gpuStepCache.CellNext!.ToCpuTensor(),
            TanhCellNext = gpuStepCache.TanhCellNext!.ToCpuTensor(),
            HiddenNext = gpuStepCache.HiddenNext!.ToCpuTensor()
        };

        _ramCache[timeStep] = ramItem;
        _lruTracker.AddFirst(timeStep);
        _diskOffsets[timeStep] = new Dictionary<string, long>(); // Prepara para armazenar offsets se for para o disco
        
        if (timeStep > _maxTimestepCached)
            _maxTimestepCached = timeStep;
    }

    public IMathTensor RetrieveTensor(int timeStep, string tensorName)
    {
        if (_ramCache.TryGetValue(timeStep, out var ramItem))
        {
            _lruTracker.Remove(timeStep);
            _lruTracker.AddFirst(timeStep);
            
            Tensor? cpuTensor = tensorName switch
            {
                "Input" => ramItem.Input, 
                "HiddenPrev" => ramItem.HiddenPrev,
                "CellPrev" => ramItem.CellPrev, 
                "ForgetGate" => ramItem.ForgetGate,
                "InputGate" => ramItem.InputGate, 
                "CellCandidate" => ramItem.CellCandidate,
                "OutputGate" => ramItem.OutputGate, 
                "CellNext" => ramItem.CellNext,
                "TanhCellNext" => ramItem.TanhCellNext, 
                "HiddenNext" => ramItem.HiddenNext,
                _ => throw new ArgumentException($"Nome do tensor inválido: {tensorName}")
            };
            return _mathEngine.CreateTensor(cpuTensor!.GetData(), cpuTensor.GetShape());
        }
        else
        {
            if (!_diskOffsets.ContainsKey(timeStep) || !_diskOffsets[timeStep].ContainsKey(tensorName))
                throw new KeyNotFoundException($"Tensor '{tensorName}' para o timestep {timeStep} não foi encontrado no cache de disco.");
                
            long offset = _diskOffsets[timeStep][tensorName];
            var shape = _tensorShapes[tensorName];
            int length = _accessor.ReadInt32(offset);
            var data = new double[length];
            _accessor.ReadArray(offset + sizeof(int), data, 0, length);

            return _mathEngine.CreateTensor(data, shape);
        }
    }

    private void EvictLruItemToDisk()
    {
        int lruTimeStep = _lruTracker.Last!.Value;
        var ramItemToEvict = _ramCache[lruTimeStep];
        
        // Escreve cada tensor do item no disco e armazena sua localização (offset)
        _diskOffsets[lruTimeStep]["Input"] = WriteTensorToDisk(ramItemToEvict.Input!);
        _diskOffsets[lruTimeStep]["HiddenPrev"] = WriteTensorToDisk(ramItemToEvict.HiddenPrev!);
        _diskOffsets[lruTimeStep]["CellPrev"] = WriteTensorToDisk(ramItemToEvict.CellPrev!);
        _diskOffsets[lruTimeStep]["ForgetGate"] = WriteTensorToDisk(ramItemToEvict.ForgetGate!);
        _diskOffsets[lruTimeStep]["InputGate"] = WriteTensorToDisk(ramItemToEvict.InputGate!);
        _diskOffsets[lruTimeStep]["CellCandidate"] = WriteTensorToDisk(ramItemToEvict.CellCandidate!);
        _diskOffsets[lruTimeStep]["OutputGate"] = WriteTensorToDisk(ramItemToEvict.OutputGate!);
        _diskOffsets[lruTimeStep]["CellNext"] = WriteTensorToDisk(ramItemToEvict.CellNext!);
        _diskOffsets[lruTimeStep]["TanhCellNext"] = WriteTensorToDisk(ramItemToEvict.TanhCellNext!);
        _diskOffsets[lruTimeStep]["HiddenNext"] = WriteTensorToDisk(ramItemToEvict.HiddenNext!);
        
        // Remove da RAM após ser salvo no disco
        _ramCache.Remove(lruTimeStep);
        _lruTracker.RemoveLast();
    }

    private long WriteTensorToDisk(Tensor tensor)
    {
        var data = tensor.GetData();
        int length = data.Length;
        int byteLength = length * sizeof(double);
        long startOffset = _currentDiskPosition;

        // Formato simples: [tamanho do array (int)] seguido por [dados do array (doubles)]
        _accessor.Write(startOffset, length);
        _accessor.WriteArray(startOffset + sizeof(int), data, 0, length);
        
        _currentDiskPosition += sizeof(int) + byteLength;
        return startOffset;
    }

    /// <summary>
    /// MÉTODO CRÍTICO: Limpa o estado do cache para o próximo lote.
    /// Esta função resolve o vazamento de memória ao limpar os metadados.
    /// </summary>
    public void Reset()
    {
        // 1. Limpa o cache de RAM
        _ramCache.Clear();
        _lruTracker.Clear();
        
        // 2. LIMPA OS METADADOS DE DISCO (A CORREÇÃO PRINCIPAL)
        _diskOffsets.Clear();
        
        // 3. Reseta a posição de escrita no arquivo de disco
        _currentDiskPosition = 0;
        _maxTimestepCached = 0;
        
        // 4. Sugere ao GC que pode ser um bom momento para uma limpeza leve
        GC.Collect(1, GCCollectionMode.Optimized, false);
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        
        // Garante que todos os recursos sejam limpos ao descartar o objeto
        Reset();
        _accessor?.Dispose();
        _mmf?.Dispose();

        try
        {
            if (File.Exists(_tempFilePath))
            {
                File.Delete(_tempFilePath);
            }
        }
        catch (IOException) 
        { 
            // Ignora erros se o arquivo não puder ser deletado, pois é temporário.
        }
        
        GC.SuppressFinalize(this);
    }
}