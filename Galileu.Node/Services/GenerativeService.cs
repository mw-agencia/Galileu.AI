using Galileu.Node.Brain;
using Galileu.Node.Core;
using Galileu.Node.Interfaces;
using Galileu.Node.Models;
using System.IO;          
using System.Text.Json;
using System.Threading.Tasks;
using Galileu.Node.Brain.Gpu;

namespace Galileu.Node.Services;

public class GenerativeService
{
    private int _inputSize;
    private int _hiddenSize;
    private int _outputSize;
    private int _contextWindowSize;
    private string _modelPath = Path.Combine(Environment.CurrentDirectory, "Dayson", "Dayson.json");
    private ISearchService _searchService = new MockSearchService();

    private readonly IMathEngine _mathEngine;
    private readonly PrimingService _primingService;
    
    private GenerativeNeuralNetworkLSTM? _model;
    
    public bool IsModelLoaded => _model != null;

    public GenerativeService(OpenCLService openCLService, PrimingService primingService)
    {
        _primingService = primingService;
        if (openCLService.IsGpuAvailable)
        {
            Console.WriteLine("[GenerativeService] Usando GpuMathEngine para aceleração.");
            _mathEngine = new GpuMathEngine(openCLService);
        }
        else
        {
            Console.WriteLine("[GenerativeService] GPU não disponível. Usando CpuMathEngine como fallback.");
            _mathEngine = new CpuMathEngine();
        }
    }

    public void InitializeFromDisk()
    {
        Console.WriteLine("[GenerativeService] Tentando inicializar o modelo do disco...");
        if (!File.Exists(_modelPath))
        {
            Console.WriteLine($"[GenerativeService] Nenhum modelo encontrado em '{_modelPath}'. O treinamento é necessário.");
            return;
        }

        // --- CORREÇÃO 2: A lógica do antigo 'TryLoadConfigurationFromModel' foi mesclada aqui ---
        // E a chamada if() redundante foi removida.
        bool configLoaded = LoadConfigurationFromModelFile();
        if (configLoaded)
        {
            _model = ModelSerializerLSTM.LoadModel(_modelPath, _mathEngine);
            if (_model != null)
            {
                Console.WriteLine("[GenerativeService] Modelo carregado com sucesso.");
                _primingService.PrimeModel(_model);
            }
            else
            {
                 Console.WriteLine("[GenerativeService] Falha ao carregar o modelo, embora o arquivo de configuração exista.");
            }
        }
    }

    public void Configure(int inputSize, int hiddenSize, int outputSize, int contextWindowSize, string modelPath, ISearchService searchService)
    {
        _inputSize = inputSize;
        _hiddenSize = hiddenSize;
        _outputSize = outputSize;
        _contextWindowSize = contextWindowSize;
        _modelPath = modelPath;
        _searchService = searchService;
        Console.WriteLine($"[GenerativeService] Configurado. InputSize: {_inputSize}, HiddenSize: {_hiddenSize}");
    }

    // Este método agora é privado e usado apenas pela inicialização.
    private bool LoadConfigurationFromModelFile()
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
                Configure(
                    modelData.InputSize,
                    modelData.HiddenSize,
                    modelData.OutputSize,
                    5, // Assumimos o mesmo valor usado no treinamento
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
        await Task.Run(() =>
        {
            GenerativeNeuralNetworkLSTM? trainingModel;

            if (File.Exists(_modelPath))
            {
                Console.WriteLine("Continuando o treinamento do modelo existente...");
                trainingModel = ModelSerializerLSTM.LoadModel(_modelPath, _mathEngine);
                if (trainingModel == null)
                {
                     Console.WriteLine("Falha ao carregar modelo para treinamento. Criando um novo.");
                     trainingModel = new GenerativeNeuralNetworkLSTM(_inputSize, _hiddenSize, _outputSize, trainerOptions.datasetPath, _searchService, _mathEngine);
                }
            }
            else
            {
                Console.WriteLine("Criando e treinando um novo modelo LSTM generativo...");
                trainingModel = new GenerativeNeuralNetworkLSTM(_inputSize, _hiddenSize, _outputSize, trainerOptions.datasetPath, _searchService, _mathEngine);
            }
            
            var trainer = new ModelTrainerLSTM(trainingModel);
            trainer.TrainModel(trainerOptions.datasetPath, trainerOptions.learningRate, trainerOptions.epochs,
                batchSize: 32, contextWindowSize: _contextWindowSize, trainerOptions.validationSplit);

            trainingModel.SaveModel(_modelPath);
            Console.WriteLine($"Modelo salvo com sucesso em: {_modelPath}. Recarregando para inferência.");
            
            InitializeFromDisk(); 
        });
    }

    public async Task<string?> GenerateAsync(GenerateResponse generateResponse)
    {
        if (_model == null)
        {
            return "Erro: O modelo não está carregado. Treine ou reinicie o serviço.";
        }

        return await Task.Run(() => _model.GenerateResponse(generateResponse.input, maxLength: 50));
    }
}