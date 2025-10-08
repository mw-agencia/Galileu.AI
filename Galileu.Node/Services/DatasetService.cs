using Galileu.Node.Core;
using Galileu.Node.TreeSwapFile;
using System.Text.Json;

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

    public void InitializeAndSplit(string text, int contextWindow,
        Dictionary<string, int> vocab, string padToken, int initialBatchSize, double validationSplit)
    {
        lock (_lock)
        {
            _batchSize = initialBatchSize;
            _storage.Clear();
            var allOffsets = new List<long>();
            var tokens = text.ToLower().Split(new[] { ' ', '\n', '\r' }, StringSplitOptions.RemoveEmptyEntries);
            int unkIndex = vocab["<UNK>"];
            int vocabSize = vocab.Count;
            int totalSamples = 0;

            for (int i = 0; i < tokens.Length - contextWindow; i++)
            {
                var inputToken = tokens[i + contextWindow - 1]; // Usamos apenas o token anterior para prever o próximo
                var targetToken = tokens[i + contextWindow];

                int inputIndex = vocab.TryGetValue(inputToken, out var idx) ? idx : unkIndex;
                var targetTensor = CreateOneHotTensor(targetToken, vocab, vocabSize);
                
                var pairData = new SampleIndexData
                {
                   InputIndex = inputIndex,
                   TargetIndex = Array.IndexOf(targetTensor.GetData(), 1.0) // Armazena o índice do alvo
                };

                string json = JsonSerializer.Serialize(pairData);
                long offset = _storage.StoreData(json);
                allOffsets.Add(offset);
                totalSamples++;
            }
            Console.WriteLine($"[DatasetService] Amostras armazenadas: {totalSamples}");

            _storage.Flush();
            Console.WriteLine($"[DatasetService] Armazenamento concluído. {allOffsets.Count} amostras válidas totais.");

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

    public List<(int InputIndex, Tensor Target)>? GetNextTrainChunk() => GetNextChunk(ref _currentTrainIndex, _trainOffsets);
    public List<(int InputIndex, Tensor Target)>? GetNextValidationChunk() => GetNextChunk(ref _currentValidationIndex, _validationOffsets);

    private List<(int InputIndex, Tensor Target)>? GetNextChunk(ref int currentIndex, List<long> offsets)
    {
        var chunk = new List<(int, Tensor)>();
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
                    double[] targetData = new double[20000]; // ATENÇÃO: Hardcoded vocab size
                    targetData[pairData.TargetIndex] = 1.0;
                    chunk.Add((
                        pairData.InputIndex,
                        new Tensor(targetData, new [] { 20000 })
                    ));
                }
            }
            currentIndex = endIndex;
        }
        return chunk;
    }

    public void ResetTrain() => _currentTrainIndex = 0;
    public void ResetValidation() => _currentValidationIndex = 0;

    private Tensor CreateOneHotTensor(string token, Dictionary<string, int> vocab, int vocabSize)
    {
        double[] data = new double[vocabSize];
        if (vocab.TryGetValue(token, out int index)) data[index] = 1.0;
        else data[vocab["<UNK>"]] = 1.0;
        return new Tensor(data, new[] { vocabSize });
    }

    public void Dispose() => _storage?.Dispose();
}