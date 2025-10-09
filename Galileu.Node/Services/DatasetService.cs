using Galileu.Node.TreeSwapFile;
using System.Text.Json;
using System.IO;
using System.Linq;
using System.Collections.Generic;
using System;

namespace Galileu.Node.Services;

public class DatasetService : IDisposable
{
    private readonly BinaryTreeFileStorage _storage;
    private List<long> _trainOffsets;
    private List<long> _validationOffsets;
    private int _currentTrainIndex = 0;
    private int _currentValidationIndex = 0;
    private int _batchSize;
    private readonly object _lock = new object();

    public DatasetService(string storageFilePath)
    {
        _storage = new BinaryTreeFileStorage(storageFilePath);
        _trainOffsets = new List<long>();
        _validationOffsets = new List<long>();
    }

    // --- CORREÇÃO: Assinatura do método atualizada para corresponder à chamada ---
    public void InitializeAndSplit(string datasetPath, int contextWindow,
        Dictionary<string, int> vocab, int initialBatchSize, double validationSplit)
    {
        if (!File.Exists(datasetPath))
            throw new FileNotFoundException("Arquivo de dataset não encontrado no DatasetService.", datasetPath);

        string text = File.ReadAllText(datasetPath);
        
        lock (_lock)
        {
            _batchSize = initialBatchSize;
            _storage.Clear();
            var allOffsets = new List<long>();
            var tokens = text.ToLower().Split(new[] { ' ', '\n', '\r' }, StringSplitOptions.RemoveEmptyEntries);
            int unkIndex = vocab["<UNK>"];
            int totalSamples = 0;

            for (int i = 0; i < tokens.Length - contextWindow -1; i++) // Ajuste no loop para evitar OutOfBounds
            {
                var inputToken = tokens[i + contextWindow - 1];
                var targetToken = tokens[i + contextWindow];

                if (vocab.TryGetValue(inputToken, out var inputIndex) && vocab.TryGetValue(targetToken, out var targetIndex))
                {
                    var pairData = new SampleIndexData
                    {
                       InputIndex = inputIndex,
                       TargetIndex = targetIndex
                    };

                    string json = JsonSerializer.Serialize(pairData);
                    long offset = _storage.StoreData(json);
                    allOffsets.Add(offset);
                    totalSamples++;
                }
            }
            Console.WriteLine($"[DatasetService] Amostras válidas armazenadas: {totalSamples}");

            _storage.Flush();
            Console.WriteLine($"[DatasetService] Armazenamento concluído. {allOffsets.Count} amostras totais.");

            var rnd = new Random();
            allOffsets = allOffsets.OrderBy(x => rnd.Next()).ToList();
            int validationCount = (int)(allOffsets.Count * validationSplit);
            _validationOffsets = allOffsets.Take(validationCount).ToList();
            _trainOffsets = allOffsets.Skip(validationCount).ToList();

            Console.WriteLine($"[DatasetService] Offsets divididos: {_trainOffsets.Count / _batchSize} lotes de treino, {_validationOffsets.Count / _batchSize} lotes de validação.");
        }
    }
    
    public void SetBatchSize(int newSize)
    {
        lock (_lock) { _batchSize = Math.Max(1, newSize); }
    }

    public List<(int InputIndex, int TargetIndex)>? GetNextTrainChunk() => GetNextChunk(ref _currentTrainIndex, _trainOffsets);
    public List<(int InputIndex, int TargetIndex)>? GetNextValidationChunk() => GetNextChunk(ref _currentValidationIndex, _validationOffsets);

    private List<(int InputIndex, int TargetIndex)>? GetNextChunk(ref int currentIndex, List<long> offsets)
    {
        var chunk = new List<(int, int)>();
        lock (_lock)
        {
            if (currentIndex >= offsets.Count) return null;
            int endIndex = Math.Min(currentIndex + _batchSize, offsets.Count);
            for (int i = currentIndex; i < endIndex; i++)
            {
                string json = _storage.GetData(offsets[i]);
                var pairData = JsonSerializer.Deserialize<SampleIndexData>(json);
                if (pairData != null)
                {
                    chunk.Add((pairData.InputIndex, pairData.TargetIndex));
                }
            }
            currentIndex = endIndex;
        }
        return chunk;
    }
    
    public int GetTotalTrainBatches()
    {
        lock (_lock)
        {
            if (_batchSize == 0) return 0;
            return (int)Math.Ceiling((double)_trainOffsets.Count / _batchSize);
        }
    }

    public int GetTotalValidationBatches()
    {
        lock (_lock)
        {
            if (_batchSize == 0) return 0;
            return (int)Math.Ceiling((double)_validationOffsets.Count / _batchSize);
        }
    }

    public void ResetTrain() => _currentTrainIndex = 0;
    public void ResetValidation() => _currentValidationIndex = 0;

    public void Dispose() => _storage?.Dispose();
}