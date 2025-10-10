using System;
using System.Collections.Generic;
using System.Linq;
using Galileu.Node.Core;
using Galileu.Node.Interfaces;

namespace Galileu.Node.Brain;

/// <summary>
/// Estende a rede neural base com a capacidade de gerar texto.
/// Compatível com a arquitetura "Totalmente Out-of-Core".
/// </summary>
public class GenerativeNeuralNetworkLSTM : NeuralNetworkLSTM
{
    private readonly VocabularyManager vocabularyManager;
    private readonly ISearchService searchService;
    private readonly int _embeddingSize;

    /// <summary>
    /// Construtor para criar um novo modelo generativo para treinamento.
    /// </summary>
    public GenerativeNeuralNetworkLSTM(int vocabSize, int embeddingSize, int hiddenSize, string datasetPath,
        ISearchService? searchService, IMathEngine mathEngine)
        : base(vocabSize, embeddingSize, hiddenSize, vocabSize, mathEngine) 
    {
        this.vocabularyManager = new VocabularyManager();
        this.searchService = searchService ?? new MockSearchService();
        this._embeddingSize = embeddingSize;

        int loadedVocabSize = vocabularyManager.BuildVocabulary(datasetPath, maxVocabSize: vocabSize);
        if (loadedVocabSize == 0)
        {
            throw new InvalidOperationException("Vocabulário vazio. Verifique o arquivo de dataset.");
        }
        if (loadedVocabSize != vocabSize)
        {
            throw new ArgumentException($"O tamanho do vocabulário construído ({loadedVocabSize}) não corresponde ao solicitado ({vocabSize}).");
        }
    }

    /// <summary>
    /// Construtor para carregar um modelo generativo existente.
    /// </summary>
    public GenerativeNeuralNetworkLSTM(string modelConfigPath, VocabularyManager vocabManager, ISearchService? searchService, IMathEngine mathEngine)
        : base(modelConfigPath, mathEngine)
    {
        this.vocabularyManager = vocabManager ?? throw new ArgumentNullException(nameof(vocabManager));
        this.searchService = searchService ?? new MockSearchService();
        this._embeddingSize = this.GetEmbeddingSizeFromModel();
    }

    private int GetEmbeddingSizeFromModel()
    {
        // Usa o _paramManager herdado para obter o peso e descobrir sua forma.
        using var w_embed = _paramManager.GetParameter("w_embed");
        return w_embed.Shape[1];
    }

    /// <summary>
    /// Gera uma continuação de texto a partir de um prompt de entrada.
    /// </summary>
    public string GenerateResponse(string inputText, int maxLength = 50)
    {
        if (string.IsNullOrEmpty(inputText)) return "Erro: Entrada vazia ou nula.";

        ResetHiddenState();
        var tokens = Tokenize(inputText);
        
        using var embeddingVectorProxy = _tensorPool!.Rent(new[] { 1, _embeddingSize });
        embeddingVectorProxy.MarkDirty();

        // Aquece o estado da rede com o prompt
        foreach (var token in tokens.Take(tokens.Length - 1))
        {
            int tokenIndex = GetTokenIndex(token);
            
            // Usa o _paramManager herdado para carregar o peso de embedding sob demanda.
            using(var w_embed = _paramManager.GetParameter("w_embed"))
            {
                GetMathEngine().Lookup(w_embed, tokenIndex, embeddingVectorProxy.GetTensor());
            }

            Forward(new Tensor(embeddingVectorProxy.GetTensor().ToCpuTensor().GetData(), new[] { _embeddingSize }));
        }

        var responseTokens = new List<string>();
        string lastToken = tokens.LastOrDefault() ?? "<UNK>";

        // Loop de geração
        for (int i = 0; i < maxLength; i++)
        {
            int lastTokenIndex = GetTokenIndex(lastToken);
            
            using(var w_embed = _paramManager.GetParameter("w_embed"))
            {
                GetMathEngine().Lookup(w_embed, lastTokenIndex, embeddingVectorProxy.GetTensor());
            }
            
            var output = Forward(new Tensor(embeddingVectorProxy.GetTensor().ToCpuTensor().GetData(), new[] { _embeddingSize })); 

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
        return response.Length > 0 ? response.Capitalize() : "Não foi possível gerar uma resposta.";
    }
    
    private int GetTokenIndex(string token)
    {
        return vocabularyManager.Vocab.TryGetValue(token, out int tokenIndex) 
            ? tokenIndex 
            : vocabularyManager.Vocab["<UNK>"];
    }

    private string[] Tokenize(string text)
    {
        return text.ToLower().Split(new[] { ' ' }, StringSplitOptions.RemoveEmptyEntries);
    }
    
    private int SampleToken(Tensor output)
    {
        double[] probs = output.GetData();
        var rand = new Random();
        double r = rand.NextDouble();
        double cumulative = 0;
        for (int i = 0; i < probs.Length; i++)
        {
            cumulative += probs[i];
            if (r <= cumulative) return i;
        }
        return probs.Length - 1;
    }

    internal VocabularyManager VocabularyManager => vocabularyManager;
}