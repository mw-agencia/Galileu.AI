// --- ARQUIVO MODIFICADO: LstmCacheManager.cs (VERSÃO 2.0 - GRANULAR) ---

using Galileu.Node.Interfaces;
using System;
using System.Collections.Generic;
using System.IO;

namespace Galileu.Node.Brain;

/// <summary>
/// Gerencia o cache em disco de tensores LSTM com carregamento granular sob demanda.
/// Armazena os offsets de cada tensor individualmente para minimizar o consumo
/// de memória durante o backward pass.
/// </summary>
public class LstmCacheManager : IDisposable
{
    private readonly IMathEngine _mathEngine;
    private readonly FileStream _fileStream;
    private readonly BinaryWriter _writer;
    private readonly BinaryReader _reader;
    // --- MUDANÇA ESTRUTURAL: Armazena um dicionário de offsets por passo ---
    private readonly List<Dictionary<string, long>> _tensorOffsets;
    private bool _disposed = false;

    // Formas (shapes) dos tensores, mapeadas por nome para recuperação
    private readonly Dictionary<string, int[]> _tensorShapes;

    // Constantes para evitar "magic strings"
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

    public LstmCacheManager(IMathEngine mathEngine, int embeddingSize, int hiddenSize)
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

        var tempFilePath = Path.Combine(Environment.CurrentDirectory, "Dayson", $"lstm_cache_.bin");
        _fileStream = new FileStream(tempFilePath, FileMode.Create, FileAccess.ReadWrite, FileShare.None, 4096, FileOptions.DeleteOnClose);
        _writer = new BinaryWriter(_fileStream);
        _reader = new BinaryReader(_fileStream);
    }

    /// <summary>
    /// Escreve os tensores no disco e armazena o offset de cada um.
    /// </summary>
    public void CacheStep(LstmStepCache stepCache)
    {
        var currentStepOffsets = new Dictionary<string, long>();

        // Escreve cada tensor e salva seu offset individual
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

        _tensorOffsets.Add(currentStepOffsets);
    }

    /// <summary>
    /// Recupera UM ÚNICO tensor do disco, especificado pelo passo de tempo e nome.
    /// </summary>
    public IMathTensor RetrieveTensor(int timeStep, string tensorName)
    {
        long offset = _tensorOffsets[timeStep][tensorName];
        int[] shape = _tensorShapes[tensorName];

        _fileStream.Seek(offset, SeekOrigin.Begin);
        
        int length = _reader.ReadInt32();
        var data = new double[length];
        // Otimização: ler bytes em bloco e depois converter
        var buffer = _reader.ReadBytes(length * sizeof(double));
        Buffer.BlockCopy(buffer, 0, data, 0, buffer.Length);
        
        return _mathEngine.CreateTensor(data, shape);
    }
    
    private void WriteAndRecordOffset(string name, IMathTensor tensor, Dictionary<string, long> offsetDict)
    {
        offsetDict[name] = _fileStream.Position;
        WriteTensorData(tensor);
    }

    private void WriteTensorData(IMathTensor tensor)
    {
        var cpuData = tensor.ToCpuTensor().GetData();
        _writer.Write(cpuData.Length);
        // Otimização: converter para bytes e escrever em bloco
        var buffer = new byte[cpuData.Length * sizeof(double)];
        Buffer.BlockCopy(cpuData, 0, buffer, 0, buffer.Length);
        _writer.Write(buffer);
    }

    public void Reset()
    {
        _fileStream.SetLength(0);
        _tensorOffsets.Clear();
    }

    public void Dispose()
    {
        if (_disposed) return;
        _writer?.Close(); // Close já chama Dispose
        _reader?.Close();
        _fileStream?.Close(); // O FileOptions.DeleteOnClose cuidará da exclusão
        _disposed = true;
        GC.SuppressFinalize(this);
    }
}