using Galileu.Node.Core;
using Galileu.Node.Interfaces;
using System;
using System.Collections.Generic;
using System.Linq;

namespace Galileu.Node.Brain;

public class GenerativeNeuralNetworkLSTM : NeuralNetworkLSTM
{
    private readonly VocabularyManager vocabularyManager;
    private readonly ISearchService searchService;

    // --- CORREÇÃO: Construtor para criar um novo modelo para treinamento ---
    // Agora aceita IMathEngine em vez de OpenCLService.
    public GenerativeNeuralNetworkLSTM(int inputSize, int hiddenSize, int outputSize, string datasetPath,
        ISearchService? searchService, IMathEngine mathEngine)
        : base(inputSize, hiddenSize, outputSize, mathEngine) // Passa a engine para a classe base
    {
        this.vocabularyManager = new VocabularyManager();
        this.searchService = searchService ?? new MockSearchService();
        int vocabSize = vocabularyManager.BuildVocabulary(datasetPath);
        if (vocabSize == 0)
        {
            throw new InvalidOperationException("Vocabulário vazio. Verifique o arquivo de dataset.");
        }

        // Validação crucial: o input da rede deve ser do tamanho do vocabulário para one-hot encoding
        if (vocabSize != inputSize)
        {
            throw new ArgumentException(
                $"O tamanho do vocabulário ({vocabSize}) deve ser igual ao inputSize ({inputSize}).");
        }

        if (vocabSize != outputSize)
        {
            throw new ArgumentException(
                $"O tamanho do vocabulário ({vocabSize}) deve ser igual ao outputSize ({outputSize}).");
        }
    }

    // --- CORREÇÃO: Novo construtor para "envolver" um modelo base já carregado ---
    // Este é o construtor que o ModelSerializerLSTM usará.
    public GenerativeNeuralNetworkLSTM(NeuralNetworkLSTM baseModel, VocabularyManager vocabManager, ISearchService? searchService)
        // Chama o construtor protegido da classe base (NeuralNetworkLSTM) para transferir
        // os pesos e a configuração que já foram carregados.
        : base(
              baseModel.InputSize, 
              baseModel.HiddenSize, 
              baseModel.OutputSize,
              // As propriedades a seguir precisam ser públicas na classe NeuralNetworkLSTM
              baseModel.WeightsInputForget, baseModel.WeightsHiddenForget,
              baseModel.WeightsInputInput, baseModel.WeightsHiddenInput,
              baseModel.WeightsInputCell, baseModel.WeightsHiddenCell,
              baseModel.WeightsInputOutput, baseModel.WeightsHiddenOutput,
              baseModel.BiasForget, baseModel.BiasInput,
              baseModel.BiasCell, baseModel.BiasOutput,
              baseModel.WeightsHiddenOutputFinal, baseModel.BiasOutputFinal,
              baseModel.GetMathEngine() // E passamos a engine que ele já está usando
          )
    {
        this.vocabularyManager = vocabManager ?? throw new ArgumentNullException(nameof(vocabManager));
        this.searchService = searchService ?? new MockSearchService();
    }
    
    // O construtor antigo que aceitava múltiplos Tensors foi REMOVIDO por ser obsoleto.

    public string GenerateResponse(string inputText, int maxLength = 50)
    {
        if (string.IsNullOrEmpty(inputText))
        {
            return "Erro: Entrada vazia ou nula.";
        }

        ResetHiddenState();
        var tokens = Tokenize(inputText);

        foreach (var token in tokens)
        {
            var inputTensor = CreateOneHotTensorForToken(token);
            Forward(inputTensor); // Aquece a rede com o prompt
        }

        List<string> responseTokens = new List<string>();
        string lastToken = tokens.LastOrDefault() ?? "<UNK>";

        for (int i = 0; i < maxLength; i++)
        {
            var inputTensor = CreateOneHotTensorForToken(lastToken);
            // O método Forward agora é acelerado por GPU, esta chamada não precisa mudar.
            var output = Forward(inputTensor); 

            int predictedTokenIndex = SampleToken(output);
            string predictedToken = vocabularyManager.ReverseVocab.ContainsKey(predictedTokenIndex)
                ? vocabularyManager.ReverseVocab[predictedTokenIndex]
                : "<UNK>";

            if (predictedToken == "." || predictedToken == "!" || predictedToken == "?")
            {
                responseTokens.Add(predictedToken);
                break;
            }

            responseTokens.Add(predictedToken);
            lastToken = predictedToken;
        }

        string response = string.Join(" ", responseTokens).Trim();
        if (string.IsNullOrEmpty(response))
        {
            response = "Não foi possível gerar uma resposta.";
        }

        return response.Capitalize();
    }

    // Métodos utilitários (não precisam de alteração)
    private string[] Tokenize(string text)
    {
        // Nota: Esta tokenização é simples. A do VocabularyManager com Regex é melhor.
        return text.ToLower().Split(new[] { ' ' }, StringSplitOptions.RemoveEmptyEntries);
    }

    private Tensor CreateOneHotTensorForToken(string token)
    {
        double[] inputData = new double[InputSize];
        if (vocabularyManager.Vocab.TryGetValue(token, out int tokenIndex))
        {
            if (tokenIndex < inputData.Length)
                inputData[tokenIndex] = 1.0;
        }
        else
        {
            if (vocabularyManager.Vocab.TryGetValue("<UNK>", out int unkIndex) && unkIndex < inputData.Length)
                inputData[unkIndex] = 1.0;
        }

        return new Tensor(inputData, new int[] { InputSize });
    }

    private int SampleToken(Tensor output)
    {
        if (output == null || output.GetData().Length == 0)
        {
            throw new InvalidOperationException("Tensor de saída inválido ou vazio.");
        }

        double[] probs = output.GetData();
        double total = probs.Sum();
        if (total < 1e-9)
        {
            // Fallback para uma escolha aleatória se a soma das probabilidades for zero.
            return new Random().Next(probs.Length);
        }

        // Normaliza as probabilidades
        for (int i = 0; i < probs.Length; i++)
        {
            probs[i] /= total;
        }

        double r = new Random().NextDouble();
        double cumulative = 0;
        for (int i = 0; i < probs.Length; i++)
        {
            cumulative += probs[i];
            if (r <= cumulative)
            {
                return i;
            }
        }

        return probs.Length - 1; // Fallback para o último token
    }

    internal VocabularyManager VocabularyManager => vocabularyManager;
}