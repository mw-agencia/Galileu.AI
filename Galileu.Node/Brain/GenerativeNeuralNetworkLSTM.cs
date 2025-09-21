using Galileu.Node.Core;
using Galileu.Node.Interfaces;
using Galileu.Node.Brain.Gpu;

namespace Galileu.Node.Brain;

public class GenerativeNeuralNetworkLSTM : NeuralNetworkLSTM
{
    private readonly VocabularyManager vocabularyManager;
    private readonly ISearchService searchService;

    public GenerativeNeuralNetworkLSTM(int inputSize, int hiddenSize, int outputSize, string datasetPath,
        ISearchService? searchService, OpenCLService? openCLService)
        : base(inputSize, hiddenSize, outputSize, openCLService)
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

    public GenerativeNeuralNetworkLSTM(int inputSize, int hiddenSize, int outputSize,
        Tensor weightsInputForget, Tensor weightsHiddenForget,
        Tensor weightsInputInput, Tensor weightsHiddenInput,
        Tensor weightsInputCell, Tensor weightsHiddenCell,
        Tensor weightsInputOutput, Tensor weightsHiddenOutput,
        Tensor biasForget, Tensor biasInput, Tensor biasCell, Tensor biasOutput,
        Tensor weightsHiddenOutputFinal, Tensor biasOutputFinal,
        VocabularyManager vocabManager,
        OpenCLService? openCLService)
        : base(inputSize, hiddenSize, outputSize,
            weightsInputForget, weightsHiddenForget, weightsInputInput, weightsHiddenInput,
            weightsInputCell, weightsHiddenCell, weightsInputOutput, weightsHiddenOutput,
            biasForget, biasInput, biasCell, biasOutput,
            weightsHiddenOutputFinal, biasOutputFinal,
            openCLService)
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

        // 1. Prepara o estado inicial da rede (priming)
        ResetHiddenState();
        var tokens = Tokenize(inputText);

        foreach (var token in tokens)
        {
            var inputTensor = CreateOneHotTensorForToken(token);
            Forward(inputTensor); // Aquece a rede com o prompt
        }

        // 2. Inicia a geração de texto
        List<string> responseTokens = new List<string>();
        string lastToken = tokens.LastOrDefault() ?? "<UNK>";

        for (int i = 0; i < maxLength; i++)
        {
            var inputTensor = CreateOneHotTensorForToken(lastToken);
            var output = Forward(inputTensor); // Gera o próximo token

            int predictedTokenIndex = SampleToken(output);
            string predictedToken = vocabularyManager.ReverseVocab.ContainsKey(predictedTokenIndex)
                ? vocabularyManager.ReverseVocab[predictedTokenIndex]
                : "<UNK>";

            // Critério de parada: se gerar um token de fim de frase ou repetir demais
            if (predictedToken == "." || predictedToken == "!" || predictedToken == "?")
            {
                responseTokens.Add(predictedToken);
                break;
            }

            responseTokens.Add(predictedToken);
            lastToken = predictedToken; // O próximo input será o token que acabamos de gerar
        }

        // 3. Pós-processamento
        string response = string.Join(" ", responseTokens).Trim();
        if (string.IsNullOrEmpty(response))
        {
            response = "Não foi possível gerar uma resposta.";
        }

        return response.Capitalize();
    }

    // Métodos utilitários refatorados
    private string[] Tokenize(string text)
    {
        return text.ToLower().Split(new[] { ' ' }, StringSplitOptions.RemoveEmptyEntries);
    }

    private Tensor CreateOneHotTensorForToken(string token)
    {
        double[] inputData = new double[InputSize]; // Cria um vetor de zeros
        if (vocabularyManager.Vocab.TryGetValue(token, out int tokenIndex))
        {
            if (tokenIndex < inputData.Length)
                inputData[tokenIndex] = 1.0; // Define a posição do token como 1
        }
        else
        {
            if (vocabularyManager.Vocab.TryGetValue("<UNK>", out int unkIndex) && unkIndex < inputData.Length)
                inputData[unkIndex] = 1.0; // Fallback para token desconhecido
        }

        return new Tensor(inputData, new int[] { InputSize });
    }

    // O resto da classe (SampleToken, etc.) permanece o mesmo...
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
            Random rand = new Random();
            return rand.Next(probs.Length);
        }

        for (int i = 0; i < probs.Length; i++)
        {
            probs[i] /= total;
        }

        Random randGen = new Random();
        double r = randGen.NextDouble();
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

    // ... os outros métodos privados (IsResponseCoherent, etc.) também permanecem
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