using System.Text.Json;
using Galileu.Node.Core;
using Galileu.Node.TreeSwapFile;
using System.Text.RegularExpressions;

namespace Galileu.Node.Services;

public class DatasetService : IDisposable
{
    // ... (propriedades e construtor permanecem os mesmos) ...
    private readonly BinaryTreeFileStorage _memoryStorage;
    private readonly string _swapFilePath;
    private List<List<long>> _trainBatchOffsets = new();
    private List<List<long>> _validationBatchOffsets = new();
    private int _currentTrainBatchIndex = 0;
    private int _currentValidationBatchIndex = 0;
    private int _vocabSize;

    public DatasetService(string swapFilePath)
    {
        _swapFilePath = swapFilePath;
        _memoryStorage = new BinaryTreeFileStorage(_swapFilePath);
    }


    public void InitializeAndSplit(
        string fullDatasetText,
        int contextWindowSize,
        Dictionary<string, int> tokenToIndex,
        string padToken,
        int batchSize,
        double validationSplit)
    {
        _memoryStorage.Clear();
        _vocabSize = tokenToIndex.Count;
        
        var allSamplesStream = StreamAllSampleIndicesAsUtf8Bytes(fullDatasetText, tokenToIndex);

        List<long> allOffsets = new List<long>();
        int count = 0;
        
        Action<int> progressCallback = c => Console.Write($"\r[DatasetService] Amostras armazenadas: {c}");
        
        foreach (var sampleBytes in allSamplesStream)
        {
            long offsetId = _memoryStorage.StoreData(sampleBytes);
            if (offsetId != -1)
            {
                allOffsets.Add(offsetId);
            }
            
            count++;
            if (count % 10000 == 0)
            {
                progressCallback(count);
            }
        }
        progressCallback(count);
        
        _memoryStorage.Flush();

        Console.WriteLine($"\n[DatasetService] Armazenamento concluído. {allOffsets.Count} amostras válidas totais.");

        // ... (o resto do método permanece o mesmo) ...
        int trainSampleCount = (int)(allOffsets.Count * (1 - validationSplit));
        var trainOffsets = allOffsets.Take(trainSampleCount).ToList();
        var validationOffsets = allOffsets.Skip(trainSampleCount).ToList();

        _trainBatchOffsets = trainOffsets.Select((offset, index) => new { offset, index }).GroupBy(x => x.index / batchSize).Select(g => g.Select(x => x.offset).ToList()).ToList();
        _validationBatchOffsets = validationOffsets.Select((offset, index) => new { offset, index }).GroupBy(x => x.index / batchSize).Select(g => g.Select(x => x.offset).ToList()).ToList();
        
        Console.WriteLine($"[DatasetService] Offsets divididos: {_trainBatchOffsets.Count} lotes de treino, {_validationBatchOffsets.Count} lotes de validação.");
    }
    
    
    private (Tensor Input, Tensor Target) ReconstructSampleFromIndices(string json)
    {
        var indexData = JsonSerializer.Deserialize<SampleIndexData>(json);
        double[] inputData = new double[_vocabSize];
        if (indexData.InputIndex >= 0) inputData[indexData.InputIndex] = 1.0;
        double[] targetData = new double[_vocabSize];
        if (indexData.TargetIndex >= 0) targetData[indexData.TargetIndex] = 1.0;
        var inputTensor = new Tensor(inputData, new int[] { _vocabSize });
        var targetTensor = new Tensor(targetData, new int[] { _vocabSize });
        return (inputTensor, targetTensor);
    }
    
    public List<(Tensor Input, Tensor Target)> GetNextTrainChunk()
    {
        if (_trainBatchOffsets.Count == 0 || _currentTrainBatchIndex >= _trainBatchOffsets.Count)
            return new List<(Tensor, Tensor)>();
        var offsetBatch = _trainBatchOffsets[_currentTrainBatchIndex++];
        return offsetBatch.Select(offset => _memoryStorage.GetData(offset)).Select(ReconstructSampleFromIndices).ToList();
    }

    public List<(Tensor Input, Tensor Target)> GetNextValidationChunk()
    {
        if (_validationBatchOffsets.Count == 0 || _currentValidationBatchIndex >= _validationBatchOffsets.Count)
            return new List<(Tensor, Tensor)>();
        var offsetBatch = _validationBatchOffsets[_currentValidationBatchIndex++];
        return offsetBatch.Select(offset => _memoryStorage.GetData(offset)).Select(ReconstructSampleFromIndices).ToList();
    }
    
    public void ResetTrain() => _currentTrainBatchIndex = 0;
    public void ResetValidation() => _currentValidationBatchIndex = 0;
    public void Dispose()
    {
        _memoryStorage?.Dispose();
        if (File.Exists(_swapFilePath)) { try { File.Delete(_swapFilePath); } catch { /* Ignora */ } }
    }


    // MUDANÇA 4: O método agora gera byte[] diretamente.
    private IEnumerable<byte[]> StreamAllSampleIndicesAsUtf8Bytes(string text, Dictionary<string, int> tokenToIndex)
    {
        string pattern = @"(\p{L}+|\p{N}+|[.,!?;:'""/\-])";
        var tokens = Regex.Matches(text.ToLower(), pattern).Select(m => m.Value).ToList();

        int unkIndex = tokenToIndex["<UNK>"];

        for (int i = 0; i < tokens.Count - 1; i++)
        {
            tokenToIndex.TryGetValue(tokens[i], out int inputIndex);
            if (inputIndex == 0 && tokens[i] != "<PAD>") inputIndex = unkIndex;
            
            tokenToIndex.TryGetValue(tokens[i + 1], out int targetIndex);
            if (targetIndex == 0 && tokens[i+1] != "<PAD>") targetIndex = unkIndex;
            
            var sampleIndex = new SampleIndexData { InputIndex = inputIndex, TargetIndex = targetIndex };

            // A mágica acontece aqui!
            yield return JsonSerializer.SerializeToUtf8Bytes(sampleIndex);
        }
    }
}