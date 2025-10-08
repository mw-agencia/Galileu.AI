using Galileu.Node.Brain;
using Galileu.Node.Core;
using Galileu.Node.Interfaces;
using Galileu.Node.Models;
using System.IO;
using System.Threading.Tasks;
using Galileu.Node.Cpu;
using Galileu.Node.Gpu;

namespace Galileu.Node.Services;

public class GenerativeService
{
    private readonly string _modelPath = Path.Combine(Environment.CurrentDirectory, "Dayson", "Dayson.json");
    private readonly ISearchService _searchService = new MockSearchService();
    private readonly IMathEngine _mathEngine;
    private readonly PrimingService _primingService;
    private GenerativeNeuralNetworkLSTM? _model;
    public bool IsModelLoaded => _model != null;

    public GenerativeService(PrimingService primingService)
    {
        _primingService = primingService;
        try
        {
            _mathEngine = new GpuMathEngine();
            Console.WriteLine("[GenerativeService] Usando GpuMathEngine para aceleração.");
        }
        catch (Exception)
        {
            _mathEngine = new CpuMathEngine();
            Console.WriteLine("[GenerativeService] Usando CpuMathEngine como fallback.");
        }
    }
    
    public async Task TrainModelAsync(Trainer trainerOptions)
    {
        if (!File.Exists(trainerOptions.datasetPath))
        {
            throw new FileNotFoundException($"Arquivo de dataset não encontrado em: {trainerOptions.datasetPath}");
        }

        await Task.Run(() =>
        {
            // --- CORREÇÃO: Parâmetros para o modelo com Embedding ---
            const int VOCAB_SIZE = 20000;
            const int EMBEDDING_SIZE = 128; // Dimensão do vetor de embedding
            const int HIDDEN_SIZE = 256;
            const int CONTEXT_WINDOW = 1; // Simplificado para prever a próxima palavra a partir da anterior

            Console.WriteLine($"[GenerativeService] Configurado. Vocab: {VOCAB_SIZE}, Embedding: {EMBEDDING_SIZE}, Hidden: {HIDDEN_SIZE}");
            
            var trainingModel = new GenerativeNeuralNetworkLSTM(VOCAB_SIZE, EMBEDDING_SIZE, HIDDEN_SIZE, trainerOptions.datasetPath, _searchService, _mathEngine);
            
            var trainer = new ModelTrainerLSTM(trainingModel);
            trainer.TrainModel(trainerOptions.datasetPath, trainerOptions.learningRate, trainerOptions.epochs,
                batchSize: trainerOptions.batchSize, // Podemos aumentar o batch size agora que é mais eficiente
                contextWindowSize: CONTEXT_WINDOW, 
                trainerOptions.validationSplit);

            // trainingModel.SaveModel(_modelPath); // O Save/Load precisa ser reescrito para o novo formato
            Console.WriteLine($"Modelo treinado. A funcionalidade de salvar foi desabilitada temporariamente.");
            
            _model = trainingModel;
            // _primingService.PrimeModel(_model); // Priming também precisa ser adaptado
        });
    }

    public async Task<string?> GenerateAsync(GenerateResponse generateResponse)
    {
        if (_model == null) return "Erro: O modelo não está carregado.";
        return await Task.Run(() => _model.GenerateResponse(generateResponse.input, maxLength: 50));
    }

    public void InitializeFromDisk()
    {
         Console.WriteLine($"[GenerativeService] Carregamento do disco desabilitado. O treinamento é necessário.");
    }
}