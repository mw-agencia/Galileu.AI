using Galileu.Node.Core;
using Galileu.Node.Interfaces;
using System;
using System.Collections.Generic;
using System.IO;

namespace Galileu.Node.Brain;

/// <summary>
/// Cache manager que GARANTE zero retenção de cache na RAM.
/// TUDO vai direto para disco, NADA fica na memória.
/// </summary>
public class DiskOnlyCacheManager : IDisposable
{
    private readonly IMathEngine _mathEngine;
    private readonly FileStream _fileStream;
    private readonly BinaryWriter _writer;
    private readonly BinaryReader _reader;
    
    // ✅ ÚNICA estrutura em RAM: Mapa de offsets (levíssimo)
    private readonly List<Dictionary<string, long>> _tensorOffsets;
    
    // ✅ Shapes pré-definidos (não mudam)
    private readonly Dictionary<string, int[]> _tensorShapes;
    
    private bool _disposed = false;
    
    // Constantes para nomes de tensores
    public static class TensorNames
    {
        public const string Input = "Input";
        public const string HiddenPrev = "HiddenPrev";
        public const string CellPrev = "CellPrev";
        public const string ForgetGate = "ForgetGate";
        public const string InputGate = "InputGate";
        public const string CellCandidate = "CellCandidate";
        public const string OutputGate = "OutputGate";
        public const string CellNext = "CellNext";
        public const string TanhCellNext = "TanhCellNext";
        public const string HiddenNext = "HiddenNext";
    }

    public DiskOnlyCacheManager(IMathEngine mathEngine, int embeddingSize, int hiddenSize)
    {
        _mathEngine = mathEngine;
        _tensorOffsets = new List<Dictionary<string, long>>();

        _tensorShapes = new Dictionary<string, int[]>
        {
            { TensorNames.Input, new[] { 1, embeddingSize } },
            { TensorNames.HiddenPrev, new[] { 1, hiddenSize } },
            { TensorNames.CellPrev, new[] { 1, hiddenSize } },
            { TensorNames.ForgetGate, new[] { 1, hiddenSize } },
            { TensorNames.InputGate, new[] { 1, hiddenSize } },
            { TensorNames.CellCandidate, new[] { 1, hiddenSize } },
            { TensorNames.OutputGate, new[] { 1, hiddenSize } },
            { TensorNames.CellNext, new[] { 1, hiddenSize } },
            { TensorNames.TanhCellNext, new[] { 1, hiddenSize } },
            { TensorNames.HiddenNext, new[] { 1, hiddenSize } }
        };

        var tempFilePath = Path.Combine(Environment.CurrentDirectory, "Dayson", $"lstm_cache_disk_only.bin");
        
        // ✅ CRÍTICO: FileOptions.DeleteOnClose garante limpeza automática
        _fileStream = new FileStream(
            tempFilePath, 
            FileMode.Create, 
            FileAccess.ReadWrite, 
            FileShare.None, 
            bufferSize: 1024 * 1024, // 1MB buffer para I/O eficiente
            FileOptions.DeleteOnClose | FileOptions.WriteThrough // Força flush imediato
        );
        
        _writer = new BinaryWriter(_fileStream);
        _reader = new BinaryReader(_fileStream);
        
        Console.WriteLine($"[DiskOnlyCache] Inicializado: {tempFilePath}");
        Console.WriteLine($"[DiskOnlyCache] Política: ZERO cache na RAM");
    }

    /// <summary>
    /// CRÍTICO: Grava tensores no disco E libera imediatamente da memória.
    /// </summary>
    public void CacheStep(LstmStepCache stepCache)
    {
        var currentStepOffsets = new Dictionary<string, long>();

        // ✅ FASE 1: Grava cada tensor no disco
        WriteAndRecordOffset(TensorNames.Input, stepCache.Input!, currentStepOffsets);
        WriteAndRecordOffset(TensorNames.HiddenPrev, stepCache.HiddenPrev!, currentStepOffsets);
        WriteAndRecordOffset(TensorNames.CellPrev, stepCache.CellPrev!, currentStepOffsets);
        WriteAndRecordOffset(TensorNames.ForgetGate, stepCache.ForgetGate!, currentStepOffsets);
        WriteAndRecordOffset(TensorNames.InputGate, stepCache.InputGate!, currentStepOffsets);
        WriteAndRecordOffset(TensorNames.CellCandidate, stepCache.CellCandidate!, currentStepOffsets);
        WriteAndRecordOffset(TensorNames.OutputGate, stepCache.OutputGate!, currentStepOffsets);
        WriteAndRecordOffset(TensorNames.CellNext, stepCache.CellNext!, currentStepOffsets);
        WriteAndRecordOffset(TensorNames.TanhCellNext, stepCache.TanhCellNext!, currentStepOffsets);
        WriteAndRecordOffset(TensorNames.HiddenNext, stepCache.HiddenNext!, currentStepOffsets);

        // ✅ FASE 2: Registra apenas os offsets (levíssimo)
        _tensorOffsets.Add(currentStepOffsets);
        
        // ✅✅✅ FASE 3: LIBERA IMEDIATAMENTE OS TENSORES GPU/RAM
        // CRÍTICO: Isso é o que estava faltando!
        stepCache.Input?.Dispose();
        stepCache.HiddenPrev?.Dispose();
        stepCache.CellPrev?.Dispose();
        stepCache.ForgetGate?.Dispose();
        stepCache.InputGate?.Dispose();
        stepCache.CellCandidate?.Dispose();
        stepCache.OutputGate?.Dispose();
        stepCache.CellNext?.Dispose();
        stepCache.TanhCellNext?.Dispose();
        stepCache.HiddenNext?.Dispose();
        
        // ✅ FASE 4: Força flush para disco (evita buffer em RAM)
        _writer.Flush();
        _fileStream.Flush(flushToDisk: true);
    }

    /// <summary>
    /// Recupera UM ÚNICO tensor do disco de forma pontual.
    /// O tensor é criado NA GPU/RAM apenas durante o uso e deve ser liberado imediatamente.
    /// </summary>
    public IMathTensor RetrieveTensor(int timeStep, string tensorName)
    {
        if (timeStep >= _tensorOffsets.Count)
        {
            throw new KeyNotFoundException($"Timestep {timeStep} não encontrado no cache.");
        }
        
        if (!_tensorOffsets[timeStep].TryGetValue(tensorName, out long offset))
        {
            throw new KeyNotFoundException($"Tensor '{tensorName}' no timestep {timeStep} não encontrado.");
        }

        int[] shape = _tensorShapes[tensorName];

        // ✅ Posiciona no offset correto
        _fileStream.Seek(offset, SeekOrigin.Begin);
        
        // ✅ Lê tamanho
        int length = _reader.ReadInt32();
        
        // ✅ Lê dados do disco direto para array
        double[] data = new double[length];
        byte[] buffer = _reader.ReadBytes(length * sizeof(double));
        Buffer.BlockCopy(buffer, 0, data, 0, buffer.Length);
        
        // ✅ Cria tensor GPU/CPU (TEMPORÁRIO - deve ser liberado pelo caller!)
        return _mathEngine.CreateTensor(data, shape);
    }
    
    /// <summary>
    /// OTIMIZAÇÃO: Recupera múltiplos tensores de uma vez (para backward pass).
    /// Agrupa I/O de disco para melhor performance.
    /// </summary>
    public Dictionary<string, IMathTensor> RetrieveMultipleTensors(int timeStep, params string[] tensorNames)
    {
        if (timeStep >= _tensorOffsets.Count)
        {
            throw new KeyNotFoundException($"Timestep {timeStep} não encontrado no cache.");
        }
        
        var tensors = new Dictionary<string, IMathTensor>();
        var offsets = _tensorOffsets[timeStep];
        
        // ✅ OTIMIZAÇÃO: Ordena por offset para leitura sequencial
        var sortedNames = tensorNames
            .Where(name => offsets.ContainsKey(name))
            .OrderBy(name => offsets[name])
            .ToArray();
        
        // ✅ Lê todos os tensores ordenados por posição no arquivo
        foreach (var name in sortedNames)
        {
            long offset = offsets[name];
            int[] shape = _tensorShapes[name];

            // Posiciona no offset correto
            _fileStream.Seek(offset, SeekOrigin.Begin);
            
            // Lê tamanho
            int length = _reader.ReadInt32();
            
            // Lê dados do disco
            double[] data = new double[length];
            byte[] buffer = _reader.ReadBytes(length * sizeof(double));
            Buffer.BlockCopy(buffer, 0, data, 0, buffer.Length);
            
            // Cria tensor GPU/CPU
            tensors[name] = _mathEngine.CreateTensor(data, shape);
        }
        
        return tensors;
    }
    
    /// <summary>
    /// OTIMIZAÇÃO: Pré-carrega próximo timestep em background (async I/O).
    /// EXPERIMENTAL: Use com cuidado, pode aumentar uso de RAM.
    /// </summary>
    public void PrefetchTimestep(int timeStep)
    {
        // TODO: Implementar leitura assíncrona
        // Isso pode acelerar backward pass em 20-30%
        // MAS adiciona complexidade e pode aumentar RAM
    }
    
    private void WriteAndRecordOffset(string name, IMathTensor tensor, Dictionary<string, long> offsetDict)
    {
        // ✅ Registra offset ANTES de escrever
        offsetDict[name] = _fileStream.Position;
        
        // ✅ Escreve dados
        WriteTensorData(tensor);
    }

    private void WriteTensorData(IMathTensor tensor)
    {
        // ✅ Converte GPU → CPU apenas para serialização
        var cpuData = tensor.ToCpuTensor().GetData();
        
        // ✅ Escreve tamanho
        _writer.Write(cpuData.Length);
        
        // ✅ Escreve dados em bloco (eficiente)
        byte[] buffer = new byte[cpuData.Length * sizeof(double)];
        Buffer.BlockCopy(cpuData, 0, buffer, 0, buffer.Length);
        _writer.Write(buffer);
    }

    /// <summary>
    /// CRÍTICO: Reseta cache para o próximo batch.
    /// Trunca arquivo para liberar espaço em disco.
    /// </summary>
    public void Reset()
    {
        // ✅ Limpa offsets
        _tensorOffsets.Clear();
        
        // ✅ Trunca arquivo (libera disco)
        _fileStream.SetLength(0);
        _fileStream.Flush(flushToDisk: true);
        
        // ✅ Reposiciona no início
        _fileStream.Seek(0, SeekOrigin.Begin);
        
        // ✅ Sugere GC (mas não força)
        if (_tensorOffsets.Capacity > 1000)
        {
            _tensorOffsets.TrimExcess();
        }
    }

    public void Dispose()
    {
        if (_disposed) return;
        
        _writer?.Dispose();
        _reader?.Dispose();
        _fileStream?.Dispose(); // DeleteOnClose cuida da exclusão
        
        _disposed = true;
        GC.SuppressFinalize(this);
    }
}