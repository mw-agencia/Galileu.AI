namespace Galileu.Node.Core;

using System;
using System.Collections.Generic;
using System.Linq;
using Galileu.Node.Interfaces;

public class GenerativeNeuralNetworkRNN : NeuralNetworkRNN
{
    private readonly VocabularyManager vocabularyManager;
    private readonly ISearchService searchService;

    public GenerativeNeuralNetworkRNN(int inputSize, int hiddenSize, int outputSize, string datasetPath, ISearchService searchService = null)
        : base(inputSize, hiddenSize, outputSize)
    {
        this.vocabularyManager = new VocabularyManager();
        this.searchService = searchService ?? new MockSearchService();
        int vocabSize = vocabularyManager.BuildVocabulary(datasetPath);
        if (vocabSize == 0)
        {
            throw new InvalidOperationException("Vocabulário vazio. Verifique o arquivo de dataset.");
        }
        if (vocabSize != outputSize)
        {
            throw new ArgumentException($"O tamanho do vocabulário ({vocabSize}) deve ser igual ao outputSize ({outputSize}).");
        }
    }

    public GenerativeNeuralNetworkRNN(int inputSize, int hiddenSize, int outputSize,
                                       Tensor weightsInputHidden, Tensor weightsHiddenHidden, Tensor biasHidden,
                                       Tensor weightsHiddenOutput, Tensor biasOutput,
                                       VocabularyManager vocabManager)
        : base(inputSize, hiddenSize, outputSize, weightsInputHidden, weightsHiddenHidden, biasHidden, weightsHiddenOutput, biasOutput)
    {
        this.vocabularyManager = vocabManager ?? throw new ArgumentNullException(nameof(vocabManager));
        this.searchService = new MockSearchService();
    }

    public string GenerateResponse(string inputText, int maxLength = 50)
    {
        if (string.IsNullOrEmpty(inputText))
        {
            return "Erro: Entrada vazia ou nula.";
        }

        ResetHiddenState();

        var tokens = inputText.Split(new[] { ' ' }, StringSplitOptions.RemoveEmptyEntries);
        if (tokens.Length == 0)
        {
            return "Erro: Nenhum token válido na entrada.";
        }

        bool needsSearch = tokens.Any(t => !vocabularyManager.Vocab.ContainsKey(t));
        string enrichedInput = inputText;
        if (needsSearch)
        {
            try
            {
                var searchResults = searchService.Search(inputText) ?? new List<string>();
                enrichedInput = $"{inputText} {string.Join(" ", searchResults)}".Trim();
                tokens = enrichedInput.Split(new[] { ' ' }, StringSplitOptions.RemoveEmptyEntries);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Erro na busca: {ex.Message}. Usando entrada original.");
            }
        }

        foreach (var token in tokens.Take(InputSize))
        {
            double[] inputData = new double[InputSize];
            if (vocabularyManager.Vocab.ContainsKey(token))
            {
                inputData[0] = vocabularyManager.Vocab[token];
            }
            else
            {
                inputData[0] = vocabularyManager.Vocab.ContainsKey("<UNK>") ? vocabularyManager.Vocab["<UNK>"] : 0;
            }
            Tensor inputTensor = new Tensor(inputData, new int[] { InputSize });
            Forward(inputTensor);
        }

        List<string> responseTokens = new List<string>();
        for (int i = 0; i < maxLength; i++)
        {
            double[] inputData = new double[InputSize];
            Tensor inputTensor = new Tensor(inputData, new int[] { InputSize });
            var output = Forward(inputTensor);
            int predictedTokenIndex = SampleToken(output);
            string predictedToken = vocabularyManager.ReverseVocab.ContainsKey(predictedTokenIndex)
                ? vocabularyManager.ReverseVocab[predictedTokenIndex]
                : "<UNK>";
            responseTokens.Add(predictedToken);
        }

        string response = string.Join(" ", responseTokens).Trim();
        if (string.IsNullOrEmpty(response))
        {
            response = "Erro: Resposta vazia gerada.";
        }
        else if (!IsResponseCoherent(inputText, response))
        {
            response = AdjustResponse(inputText, response);
        }

        return response.Capitalize();
    }

    public string GenerateSummary(string inputText, int summaryLength = 10)
    {
        if (string.IsNullOrEmpty(inputText))
        {
            return "Erro: Entrada vazia ou nula.";
        }

        ResetHiddenState();

        var tokens = inputText.Split(new[] { ' ' }, StringSplitOptions.RemoveEmptyEntries);
        if (tokens.Length == 0)
        {
            return "Erro: Nenhum token válido na entrada.";
        }

        bool needsSearch = tokens.Any(t => !vocabularyManager.Vocab.ContainsKey(t));
        string enrichedInput = inputText;
        if (needsSearch)
        {
            try
            {
                var searchResults = searchService.Search(inputText) ?? new List<string>();
                enrichedInput = $"{inputText} {string.Join(" ", searchResults)}".Trim();
                tokens = enrichedInput.Split(new[] { ' ' }, StringSplitOptions.RemoveEmptyEntries);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Erro na busca: {ex.Message}. Usando entrada original.");
            }
        }

        foreach (var token in tokens.Take(InputSize))
        {
            double[] inputData = new double[InputSize];
            if (vocabularyManager.Vocab.ContainsKey(token))
            {
                inputData[0] = vocabularyManager.Vocab[token];
            }
            else
            {
                inputData[0] = vocabularyManager.Vocab.ContainsKey("<UNK>") ? vocabularyManager.Vocab["<UNK>"] : 0;
            }
            Tensor inputTensor = new Tensor(inputData, new int[] { InputSize });
            Forward(inputTensor);
        }

        List<string> summaryTokens = new List<string>();
        for (int i = 0; i < summaryLength; i++)
        {
            double[] inputData = new double[InputSize];
            Tensor inputTensor = new Tensor(inputData, new int[] { InputSize });
            var output = Forward(inputTensor);
            int predictedTokenIndex = SampleToken(output);
            string predictedToken = vocabularyManager.ReverseVocab.ContainsKey(predictedTokenIndex)
                ? vocabularyManager.ReverseVocab[predictedTokenIndex]
                : "<UNK>";
            if (tokens.Contains(predictedToken) || predictedToken != "<UNK>")
            {
                summaryTokens.Add(predictedToken);
            }
        }

        string summary = string.Join(" ", summaryTokens).Trim();
        if (string.IsNullOrEmpty(summary))
        {
            summary = "Erro: Resumo vazio gerado.";
        }
        else if (!IsResponseCoherent(inputText, summary))
        {
            summary = AdjustResponse(inputText, summary);
        }

        return summary.Capitalize();
    }

    private int SampleToken(Tensor output)
    {
        if (output == null || output.GetData().Length == 0)
        {
            throw new InvalidOperationException("Tensor de saída inválido ou vazio.");
        }

        double[] probs = output.GetData();
        double total = probs.Sum();
        if (total == 0)
        {
            total = 1e-10;
        }
        for (int i = 0; i < probs.Length; i++)
        {
            probs[i] /= total;
        }

        Random rand = new Random();
        double r = rand.NextDouble();
        double cumulative = 0;
        for (int i = 0; i < probs.Length; i++)
        {
            cumulative += probs[i];
            if (r <= cumulative)
            {
                return i;
            }
        }
        return probs.Length - 1;
    }

    private bool IsResponseCoherent(string input, string response)
    {
        if (string.IsNullOrEmpty(input) || string.IsNullOrEmpty(response))
        {
            return false;
        }
        var inputTokens = input.Split(' ').ToHashSet();
        var responseTokens = response.Split(' ');
        return responseTokens.Any(t => inputTokens.Contains(t));
    }

    private string AdjustResponse(string input, string response)
    {
        var inputTokens = input.Split(' ').Where(t => vocabularyManager.Vocab.ContainsKey(t)).ToList();
        if (inputTokens.Any())
        {
            Random rand = new Random();
            string relevantToken = inputTokens[rand.Next(inputTokens.Count)];
            return $"{relevantToken} {response}".Trim();
        }
        return response;
    }

    internal VocabularyManager VocabularyManager => vocabularyManager;
}