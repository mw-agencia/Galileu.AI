// --- START OF FILE DatasetService.cs ---

using System.Text.Json;
using Galileu.Node.Core;
using Galileu.Node.TreeSwapFile;

namespace Galileu.Node.Services;

public class DatasetService : IDisposable
{
    private BinaryTreeFileStorage _memoryStorage;
    private List<List<long>> _batchOffsets = new();
    private int _currentBatchIndex = 0;
    private readonly string _swapFilePath;
    private List<List<long>> _trainBatchOffsets = new();
    private List<List<long>> _validationBatchOffsets = new();
    private int _currentTrainBatchIndex = 0;
    private int _currentValidationBatchIndex = 0;

    public DatasetService()
    {
        _swapFilePath = Path.Combine(Environment.CurrentDirectory, "Dayson", "memory.bin");
    }
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
        var allSamplesStream = StreamAllSamplesAsJson(fullDatasetText, contextWindowSize, tokenToIndex, padToken).ToList();
        
        // --- CORREÇÃO DEFINITIVA DA CHAMADA ---
        List<long> allOffsets;
        int count = 0;
        Action<int> progressCallback = c => Console.Write($"\r[DatasetService] Amostras armazenadas: {c}");

        // 1. Abre o arquivo uma vez para a operação de escrita completa.
        using (var fileStream = new FileStream(_swapFilePath, FileMode.Create, FileAccess.Write, FileShare.None))
        {
            // 2. Itera sobre as amostras, chamando o método correto 'StoreData' para cada uma.
            allOffsets = allSamplesStream
                .Select(sampleJson =>
                {
                    count++;
                    progressCallback(count);
                    return _memoryStorage.StoreData(sampleJson, fileStream);
                })
                .ToList();
        } // 3. O arquivo é fechado automaticamente aqui, liberando o lock.

        Console.WriteLine($"\n[DatasetService] Armazenamento concluído. {allOffsets.Count} amostras totais.");

        // 4. A lógica de divisão dos offsets (ponteiros) continua a mesma.
        int trainSampleCount = (int)(allOffsets.Count * (1 - validationSplit));
        var trainOffsets = allOffsets.Take(trainSampleCount).ToList();
        var validationOffsets = allOffsets.Skip(trainSampleCount).ToList();

        for (int i = 0; i < trainOffsets.Count; i += batchSize)
        {
            _trainBatchOffsets.Add(trainOffsets.Skip(i).Take(batchSize).ToList());
        }
        for (int i = 0; i < validationOffsets.Count; i += batchSize)
        {
            _validationBatchOffsets.Add(validationOffsets.Skip(i).Take(batchSize).ToList());
        }
        
        Console.WriteLine($"[DatasetService] Offsets divididos: {_trainBatchOffsets.Count} lotes de treino, {_validationBatchOffsets.Count} lotes de validação.");
    }
    
    // O resto da classe (GetNextTrainChunk, Dispose, etc.) permanece o mesmo.
    #region Métodos de Suporte

    public List<(Tensor Input, Tensor Target)> GetNextTrainChunk()
    {
        if (_trainBatchOffsets.Count == 0 || _currentTrainBatchIndex >= _trainBatchOffsets.Count) return new List<(Tensor, Tensor)>();
        
        var offsetBatch = _trainBatchOffsets[_currentTrainBatchIndex++];
        return offsetBatch
            .Select(offset =>
            {
                var json = _memoryStorage.GetData(offset);
                var data = JsonSerializer.Deserialize<TensorPairData>(json);
                return (new Tensor(data.Input.data, data.Input.shape), new Tensor(data.Target.data, data.Target.shape));
            })
            .ToList();
    }

    public List<(Tensor Input, Tensor Target)> GetNextValidationChunk()
    {
        if (_validationBatchOffsets.Count == 0 || _currentValidationBatchIndex >= _validationBatchOffsets.Count) return new List<(Tensor, Tensor)>();

        var offsetBatch = _validationBatchOffsets[_currentValidationBatchIndex++];
        return offsetBatch
            .Select(offset =>
            {
                var json = _memoryStorage.GetData(offset);
                var data = JsonSerializer.Deserialize<TensorPairData>(json);
                return (new Tensor(data.Input.data, data.Input.shape), new Tensor(data.Target.data, data.Target.shape));
            })
            .ToList();
    }
    
    public void ResetTrain() => _currentTrainBatchIndex = 0;
    public void ResetValidation() => _currentValidationBatchIndex = 0;

    public void Dispose()
    {
        _memoryStorage?.Dispose();
        if (File.Exists(_swapFilePath))
        {
            try { File.Delete(_swapFilePath); } catch { /* Ignora */ }
        }
    }
    
    private IEnumerable<string> StreamAllSamplesAsJson(string text, int contextWindowSize, Dictionary<string, int> tokenToIndex, string padToken)
    {
        var paddedTokens = TokenizeAndPad(text, contextWindowSize, padToken);
        int totalWindows = paddedTokens.Count - contextWindowSize;

        for (int i = 0; i < totalWindows; i++)
        {
            var (inputData, targetIndex) = CreateSampleData(paddedTokens, i, contextWindowSize, tokenToIndex);
            
            if (inputData != null)
            {
                int inputSize = contextWindowSize * tokenToIndex.Count;
                var inputTensor = new Tensor(inputData, new int[] { inputSize });
                
                int outputSize = tokenToIndex.Count;
                var targetData = new double[outputSize];
                targetData[targetIndex] = 1.0;
                var targetTensor = new Tensor(targetData, new int[] { outputSize });
                
                var sampleData = new TensorPairData
                {
                    Input = new TensorData { data = inputTensor.GetData(), shape = inputTensor.GetShape() },
                    Target = new TensorData { data = targetTensor.GetData(), shape = targetTensor.GetShape() }
                };
                
                yield return JsonSerializer.Serialize(sampleData);
            }
        }
    }
    
    private (double[]? inputData, int targetIndex) CreateSampleData(
        List<string> paddedTokens, 
        int index, 
        int contextWindowSize, 
        Dictionary<string, int> tokenToIndex)
    {
        string nextToken = paddedTokens[index + contextWindowSize];
        if (!tokenToIndex.TryGetValue(nextToken, out int targetIndex)) return (null, -1);

        int vocabSize = tokenToIndex.Count;
        int inputSize = contextWindowSize * vocabSize;
        double[] inputData = new double[inputSize];
        
        for (int k = 0; k < contextWindowSize; k++)
        {
            string token = paddedTokens[index + k];
            if (!tokenToIndex.TryGetValue(token, out int tokenVocabIndex))
            {
                tokenVocabIndex = tokenToIndex["<UNK>"];
            }
            inputData[k * vocabSize + tokenVocabIndex] = 1.0;
        }

        return (inputData, targetIndex);
    }
    
    private List<string> TokenizeAndPad(string text, int contextWindowSize, string padToken)
    {
        string pattern = @"(\p{L}+|\p{N}+|[.,!?;:'""/\-])";
        var tokens = System.Text.RegularExpressions.Regex.Matches(text.ToLower(), pattern)
            .Select(m => m.Value)
            .ToList();
        
        var paddedTokens = new List<string>(tokens.Count + contextWindowSize);
        for (int k = 0; k < contextWindowSize; k++) { paddedTokens.Add(padToken); }
        paddedTokens.AddRange(tokens);
        return paddedTokens;
    }
    #endregion
}