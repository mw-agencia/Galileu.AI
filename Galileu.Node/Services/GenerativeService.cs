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
            // === CONFIGURAÇÃO OTIMIZADA PARA 20 DIAS ===
            const int VOCAB_SIZE = 20000;
            const int EMBEDDING_SIZE = 128;  // Mantido para qualidade
            const int HIDDEN_SIZE = 256;     // Mantido para capacidade
            const int CONTEXT_WINDOW = 1;    // Simplificado para economizar memória

            Console.WriteLine($"[GenerativeService] Arquitetura: Vocab={VOCAB_SIZE}, Emb={EMBEDDING_SIZE}, Hidden={HIDDEN_SIZE}");
            
            var trainingModel = new GenerativeNeuralNetworkLSTM(
                VOCAB_SIZE, 
                EMBEDDING_SIZE, 
                HIDDEN_SIZE, 
                trainerOptions.datasetPath, 
                _searchService, 
                _mathEngine
            );
            
            // === CALCULA MEMÓRIA ESTIMADA DO MODELO ===
            long modelMemoryMB = CalculateModelMemoryMB(VOCAB_SIZE, EMBEDDING_SIZE, HIDDEN_SIZE);
            Console.WriteLine($"[GenerativeService] Memória estimada do modelo: ~{modelMemoryMB}MB");
            Console.WriteLine($"[GenerativeService] Memória disponível para cache/pool: ~{10000 - modelMemoryMB - 2000}MB");
            
            var trainer = new ModelTrainerLSTM(trainingModel);
            
            Console.WriteLine("\n[GenerativeService] Iniciando treinamento com otimizações de memória...");
            
            trainer.TrainModel(
                trainerOptions.datasetPath, 
                trainerOptions.learningRate, 
                trainerOptions.epochs,
                batchSize: trainerOptions.batchSize,
                contextWindowSize: CONTEXT_WINDOW, 
                trainerOptions.validationSplit
            );

            // === SALVA MODELO FINAL ===
            Console.WriteLine($"\n[GenerativeService] Salvando modelo final em {_modelPath}...");
            trainingModel.SaveModel(_modelPath);
            Console.WriteLine($"[GenerativeService] Modelo salvo com sucesso!");
            
            _model = trainingModel;
            
            // === LIMPEZA PÓS-TREINAMENTO ===
            Console.WriteLine("[GenerativeService] Executando limpeza pós-treinamento...");
            GC.Collect(2, GCCollectionMode.Forced, true, true);
            GC.WaitForPendingFinalizers();
        });
    }

    /// <summary>
    /// NOVO: Calcula memória aproximada do modelo em MB.
    /// </summary>
    private long CalculateModelMemoryMB(int vocabSize, int embeddingSize, int hiddenSize)
    {
        long totalParams = 0;
        
        // Embedding layer
        totalParams += vocabSize * embeddingSize;
        
        // LSTM weights (4 gates × input + hidden)
        totalParams += 4 * (embeddingSize * hiddenSize);  // Input weights
        totalParams += 4 * (hiddenSize * hiddenSize);     // Hidden weights
        totalParams += 4 * hiddenSize;                     // Biases
        
        // Output layer
        totalParams += hiddenSize * vocabSize;
        totalParams += vocabSize;
        
        // Cada parâmetro: 8 bytes (double) + Adam state (2× double) = 24 bytes
        long bytesPerParam = 24;
        long totalBytes = totalParams * bytesPerParam;
        
        return totalBytes / (1024 * 1024);
    }

    public async Task<string?> GenerateAsync(GenerateResponse generateResponse)
    {
        if (_model == null) return "Erro: O modelo não está carregado.";
        return await Task.Run(() => _model.GenerateResponse(generateResponse.input, maxLength: 50));
    }

    public void InitializeFromDisk()
    {
        if (!File.Exists(_modelPath))
        {
            Console.WriteLine($"[GenerativeService] Modelo não encontrado em {_modelPath}");
            return;
        }
        
        try
        {
            Console.WriteLine($"[GenerativeService] Carregando modelo de {_modelPath}...");
            _model = ModelSerializerLSTM.LoadModel(_modelPath, _mathEngine);
            
            if (_model != null)
            {
                Console.WriteLine("[GenerativeService] Modelo carregado com sucesso!");
            }
            else
            {
                Console.WriteLine("[GenerativeService] Falha ao carregar modelo.");
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[GenerativeService] Erro ao carregar modelo: {ex.Message}");
        }
    }
}