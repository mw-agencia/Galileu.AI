// --- START OF FILE GenerativeService.cs ---

using Galileu.Node.Brain;
using Galileu.Node.Core; // Para VocabularyManager
using Galileu.Node.Interfaces;
using Galileu.Node.Models;
using System.Text.Json; // Para JsonSerializer

namespace Galileu.Node.Services;

public class GenerativeService
{
    private int _inputSize;
    private int _hiddenSize;
    private int _outputSize;
    private int _contextWindowSize;
    private string _modelPath;
    private ISearchService _searchService;
    
    public bool IsConfigured { get; private set; } = false;

    public void Configure(int inputSize, int hiddenSize, int outputSize, int contextWindowSize, string modelPath, ISearchService searchService)
    {
        _inputSize = inputSize;
        _hiddenSize = hiddenSize;
        _outputSize = outputSize;
        _contextWindowSize = contextWindowSize;
        _modelPath = modelPath;
        _searchService = searchService;
        IsConfigured = true;
        Console.WriteLine($"[GenerativeService] Configurado com sucesso. InputSize: {_inputSize}");
    }

    public bool TryLoadConfigurationFromModel()
    {
        if (!File.Exists(_modelPath)) return false;

        try
        {
            var jsonString = File.ReadAllText(_modelPath);
            var modelData = JsonSerializer.Deserialize<NeuralNetworkModelDataLSTM>(jsonString);
            
            var vocabManager = new VocabularyManager();
            var vocabSize = vocabManager.LoadVocabulary();

            if (modelData != null && vocabSize == modelData.OutputSize)
            {
                int contextWindowSize = 5; // Assumimos o mesmo valor usado no treinamento
                Configure(
                    modelData.InputSize, 
                    modelData.HiddenSize, 
                    modelData.OutputSize,
                    contextWindowSize,
                    _modelPath,
                    new MockSearchService()
                );
                Console.WriteLine("[GenerativeService] Configuração carregada a partir de um modelo salvo.");
                return true;
            }
            return false;
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[GenerativeService] Falha ao carregar configuração do modelo: {ex.Message}");
            return false;
        }
    }

    public async Task TrainModelAsync(Trainer trainerOptions)
    {
        // (O resto do código permanece o mesmo, está correto)
        if (trainerOptions == null) return;
        
        await Task.Run(() =>
        {
            try
            {
                Console.WriteLine("Criando e treinando o modelo LSTM generativo...");
                var model = new GenerativeNeuralNetworkLSTM(_inputSize, _hiddenSize, _outputSize, trainerOptions.datasetPath, _searchService);
                
                var trainer = new ModelTrainerLSTM(model);
                trainer.TrainModel(trainerOptions.datasetPath, trainerOptions.learningRate, trainerOptions.epochs, batchSize: 32, _contextWindowSize);
                
                ModelSerializerLSTM.SaveModel(model, _modelPath);
                Console.WriteLine($"Modelo salvo com sucesso em: {_modelPath}");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Erro durante o treinamento: {ex.Message}\n{ex.StackTrace}");
            }
        });
    }

    public async Task<string?> GenerateAsync(GenerateResponse generateResponse)
    {
        // (O resto do código permanece o mesmo)
        return await Task.Run(() =>
        {
            try
            {
                var loadedModel = ModelSerializerLSTM.LoadModel(_modelPath);
                if (loadedModel == null)
                {
                    return "Erro: Modelo não pôde ser carregado.";
                }
                string response = loadedModel.GenerateResponse(generateResponse.input, maxLength: 50);
                return response;
            }
            catch (Exception ex)
            {
                return $"Erro interno durante a geração: {ex.Message}";
            }
        });
    }
}