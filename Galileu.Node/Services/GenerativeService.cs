using Galileu.Node.Brain;
using Galileu.Node.Core;
using Galileu.Node.Interfaces;
using Galileu.Node.Models;
using System.Text.Json;
using Galileu.Node.Brain.Gpu;

namespace Galileu.Node.Services;

public class GenerativeService
{
    private int _inputSize;
    private int _hiddenSize;
    private int _outputSize;
    private int _contextWindowSize;
    private string _modelPath = "";
    private ISearchService _searchService = new MockSearchService();

    // CORREÇÃO: Adicionamos o OpenCLService como uma dependência do serviço.
    private readonly OpenCLService _openCLService;

    public bool IsConfigured { get; private set; } = false;

    // CORREÇÃO: Injetamos o OpenCLService através do construtor.
    public GenerativeService(OpenCLService openCLService)
    {
        _openCLService = openCLService;
    }

    public void Configure(int inputSize, int hiddenSize, int outputSize, int contextWindowSize, string modelPath,
        ISearchService searchService)
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
        if (string.IsNullOrEmpty(_modelPath) || !File.Exists(_modelPath)) return false;

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
        if (trainerOptions == null) return;

        await Task.Run(() =>
        {
            // CORREÇÃO: Adicionando um try-catch detalhado para capturar qualquer exceção.
            try
            {
                Console.WriteLine("Criando e treinando o modelo LSTM generativo...");

                var model = new GenerativeNeuralNetworkLSTM(
                    _inputSize,
                    _hiddenSize,
                    _outputSize,
                    trainerOptions.datasetPath,
                    _searchService,
                    _openCLService);

                var trainer = new ModelTrainerLSTM(model);
                trainer.TrainModel(trainerOptions.datasetPath, trainerOptions.learningRate, trainerOptions.epochs,
                    batchSize: 32, contextWindowSize: _contextWindowSize);

                model.SaveModel(_modelPath);
                Console.WriteLine($"Modelo salvo com sucesso em: {_modelPath}");
            }
            catch (Exception ex)
            {
                // Este bloco agora irá capturar e imprimir qualquer erro que ocorra
                // em ModelTrainerLSTM ou em qualquer lugar dentro deste Task.
                Console.WriteLine("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!");
                Console.WriteLine("!!  UM ERRO CRÍTICO OCORREU DURANTE O TREINAMENTO   !!");
                Console.WriteLine("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!");
                Console.WriteLine($"Tipo de Erro: {ex.GetType().Name}");
                Console.WriteLine($"Mensagem: {ex.Message}");
                Console.WriteLine("Stack Trace:");
                Console.WriteLine(ex.StackTrace);
                Console.WriteLine("--------------------------------------------------------");
            }
        });
    }

    public async Task<string?> GenerateAsync(GenerateResponse generateResponse)
    {
        return await Task.Run(() =>
        {
            try
            {
                // CORREÇÃO: Passamos o _openCLService para o método LoadModel, que agora espera 2 argumentos.
                var loadedModel = ModelSerializerLSTM.LoadModel(_modelPath, _openCLService);
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