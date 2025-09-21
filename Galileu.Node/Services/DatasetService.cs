using System.Text.Json;
using Galileu.Node.Core;
using Galileu.Node.TreeSwapFile;
using System.Text.RegularExpressions;

namespace Galileu.Node.Services;

public class DatasetService : IDisposable
{
    private readonly BinaryTreeFileStorage _memoryStorage;
    private readonly string _swapFilePath;
    private List<List<long>> _trainBatchOffsets = new();
    private List<List<long>> _validationBatchOffsets = new();
    private int _currentTrainBatchIndex = 0;
    private int _currentValidationBatchIndex = 0;

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
        var allSamplesStream =
            StreamAllSamplesAsJson(fullDatasetText, contextWindowSize, tokenToIndex, padToken).ToList();

        List<long> allOffsets;
        int count = 0;
        Action<int> progressCallback = c => Console.Write($"\r[DatasetService] Amostras armazenadas: {c}");

        using (var fileStream = new FileStream(_swapFilePath, FileMode.Create, FileAccess.Write, FileShare.None))
        {
            allOffsets = allSamplesStream
                .Select(sampleJson =>
                {
                    count++;
                    progressCallback(count);
                    // A chamada original aqui estava incorreta, a API de BinaryTreeFileStorage
                    // espera o stream como parâmetro.
                    return _memoryStorage.StoreData(sampleJson, fileStream);
                })
                .ToList();
        }

        Console.WriteLine($"\n[DatasetService] Armazenamento concluído. {allOffsets.Count} amostras totais.");

        int trainSampleCount = (int)(allOffsets.Count * (1 - validationSplit));
        var trainOffsets = allOffsets.Take(trainSampleCount).ToList();
        var validationOffsets = allOffsets.Skip(trainSampleCount).ToList();

        _trainBatchOffsets = trainOffsets
            .Select((offset, index) => new { offset, index })
            .GroupBy(x => x.index / batchSize)
            .Select(g => g.Select(x => x.offset).ToList())
            .ToList();

        _validationBatchOffsets = validationOffsets
            .Select((offset, index) => new { offset, index })
            .GroupBy(x => x.index / batchSize)
            .Select(g => g.Select(x => x.offset).ToList())
            .ToList();

        Console.WriteLine(
            $"[DatasetService] Offsets divididos: {_trainBatchOffsets.Count} lotes de treino, {_validationBatchOffsets.Count} lotes de validação.");
    }

    public List<(Tensor Input, Tensor Target)> GetNextTrainChunk()
    {
        if (_trainBatchOffsets.Count == 0 || _currentTrainBatchIndex >= _trainBatchOffsets.Count)
            return new List<(Tensor, Tensor)>();

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
        if (_validationBatchOffsets.Count == 0 || _currentValidationBatchIndex >= _validationBatchOffsets.Count)
            return new List<(Tensor, Tensor)>();

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
            try
            {
                File.Delete(_swapFilePath);
            }
            catch
            {
                /* Ignora */
            }
        }
    }

    // CORREÇÃO: A lógica de criação de amostras foi totalmente alterada para one-hot encoding
    private IEnumerable<string> StreamAllSamplesAsJson(string text, int contextWindowSize,
        Dictionary<string, int> tokenToIndex, string padToken)
    {
        string pattern = @"(\p{L}+|\p{N}+|[.,!?;:'""/\-])";
        var tokens = Regex.Matches(text.ToLower(), pattern)
            .Select(m => m.Value)
            .ToList();

        int vocabSize = tokenToIndex.Count;

        for (int i = 0; i < tokens.Count - 1; i++)
        {
            string currentToken = tokens[i];
            string nextToken = tokens[i + 1];

            // Cria o vetor de entrada (one-hot para o token atual)
            double[] inputData = new double[vocabSize];
            if (tokenToIndex.TryGetValue(currentToken, out int inputIndex))
            {
                inputData[inputIndex] = 1.0;
            }
            else
            {
                inputData[tokenToIndex["<UNK>"]] = 1.0;
            }

            // Cria o vetor de alvo (one-hot para o próximo token)
            double[] targetData = new double[vocabSize];
            if (tokenToIndex.TryGetValue(nextToken, out int targetIndex))
            {
                targetData[targetIndex] = 1.0;
            }
            else
            {
                targetData[tokenToIndex["<UNK>"]] = 1.0;
            }

            var inputTensor = new Tensor(inputData, new int[] { vocabSize });
            var targetTensor = new Tensor(targetData, new int[] { vocabSize });

            var sampleData = new TensorPairData
            {
                Input = new TensorData { data = inputTensor.GetData(), shape = inputTensor.GetShape() },
                Target = new TensorData { data = targetTensor.GetData(), shape = targetTensor.GetShape() }
            };

            yield return JsonSerializer.Serialize(sampleData);
        }
    }
}