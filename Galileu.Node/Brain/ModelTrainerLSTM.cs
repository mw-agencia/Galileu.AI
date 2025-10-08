using System;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Threading;
using Galileu.Node.Core;
using Galileu.Node.Services;

namespace Galileu.Node.Brain;

/// <summary>
/// Orquestra o processo de treinamento de um modelo GenerativeNeuralNetworkLSTM.
/// Esta versão é otimizada para a arquitetura de embedding, passando índices de token
/// para o modelo em vez de tensores one-hot.
/// </summary>
public class ModelTrainerLSTM
{
    public readonly GenerativeNeuralNetworkLSTM model;
    private readonly Stopwatch _stopwatch = new Stopwatch();
    private GpuLoadMonitor? _gpuMonitor;

    public ModelTrainerLSTM(GenerativeNeuralNetworkLSTM model)
    {
        this.model = model ?? throw new ArgumentNullException(nameof(model));
    }

    public void TrainModel(string datasetPath, double learningRate, int epochs, int batchSize,
        int contextWindowSize, double validationSplit)
    {
        if (model.GetMathEngine().IsGpu)
        {
            _gpuMonitor = new GpuLoadMonitor(initialBatchSize: batchSize);
            Console.WriteLine("[Trainer] Monitor de GPU ativado para controle adaptativo de carga.");
        }

        if (!File.Exists(datasetPath))
            throw new FileNotFoundException("Arquivo de dataset não encontrado.", datasetPath);

        string datasetText = File.ReadAllText(datasetPath);
        if (string.IsNullOrWhiteSpace(datasetText))
            throw new InvalidOperationException("O arquivo de dataset está vazio.");

        var swapFilePath = Path.Combine(Environment.CurrentDirectory, "Dayson", "memory.bin");
        Console.WriteLine($"[Trainer] Dataset service: {swapFilePath}");

        using (var datasetService = new DatasetService(swapFilePath))
        {
            datasetService.InitializeAndSplit(datasetText, contextWindowSize,
                model.VocabularyManager.Vocab, "<PAD>", batchSize, validationSplit);

            if (_gpuMonitor != null)
            {
                datasetService.SetBatchSize(_gpuMonitor.CurrentBatchSize);
            }

            Console.WriteLine($"\n[Trainer] Configuração:");
            Console.WriteLine($"  - Épocas: {epochs}");
            Console.WriteLine($"  - Learning Rate: {learningRate}");
            Console.WriteLine($"  - Batch Size: {_gpuMonitor?.CurrentBatchSize ?? batchSize} (adaptativo se GPU)");
            Console.WriteLine($"  - Validação Split: {validationSplit:P0}\n");

            TimeSpan totalElapsedTime = TimeSpan.Zero;

            for (int epoch = 0; epoch < epochs; epoch++)
            {
                _stopwatch.Restart();
                Console.WriteLine($"\n{'═',60}");
                Console.WriteLine($"ÉPOCA {epoch + 1}/{epochs} >> {DateTime.UtcNow}");
                Console.WriteLine($"{'═',60}");

                double totalEpochLoss = 0;
                int batchCount = 0;
                datasetService.ResetTrain();
                var batchStopwatch = Stopwatch.StartNew();

                while (true)
                {
                    // OBTÉM LOTE DE ÍNDICES E TENSORES-ALVO
                    var batch = datasetService.GetNextTrainChunk();
                    if (batch == null || batch.Count == 0) break;

                    // Extrai os índices e os tensores-alvo do lote
                    var sequenceInputIndices = batch.Select(p => p.InputIndex).ToArray();
                    var sequenceTargets = batch.Select(p => p.Target).ToArray();

                    model.ResetHiddenState();

                    batchStopwatch.Restart();
                    // CHAMA O MÉTODO DE TREINAMENTO OTIMIZADO com índices
                    totalEpochLoss += model.TrainSequence(sequenceInputIndices, sequenceTargets, learningRate);
                    batchStopwatch.Stop();

                    batchCount++;
                    Console.Write($"\r Epoca : {epoch +1} / {epochs} = Total de Lotes Processados: {batchCount} ====");

                    // A lógica de monitoramento e feedback permanece a mesma
                    if (_gpuMonitor != null && batchCount % 5 == 0)
                    {
                        double gpuUtil = GpuLoadMonitor.MeasureGpuUtilization();
                        _gpuMonitor.RecordUtilization(gpuUtil, batchStopwatch.Elapsed.TotalSeconds);
                        _gpuMonitor.ApplyThrottle();
                        datasetService.SetBatchSize(_gpuMonitor.CurrentBatchSize);
                    }
                    if (batchCount % 10 == 0)
                    {
                        double avgLossSoFar = totalEpochLoss / batchCount;
                        Console.WriteLine($" \t [Época {epoch + 1}] Lotes: {batchCount} | Perda: {avgLossSoFar:F4}");
                         
                        /*
                         if (_gpuMonitor != null && batchCount % 200 == 0)
                        {
                            Console.WriteLine();
                            _gpuMonitor.PrintStatus();
                        }
                         */
                    }
                }

                _stopwatch.Stop();
                totalElapsedTime += _stopwatch.Elapsed;
                double avgLoss = batchCount > 0 ? totalEpochLoss / batchCount : double.PositiveInfinity;
                Console.WriteLine($"\n[Época {epoch + 1}] Treino concluído em {_stopwatch.Elapsed:hh\\:mm\\:ss}. Perda Média: {avgLoss:F4}");

                // Validação a cada 5 épocas
                if ((epoch + 1) % 5 == 0 || epoch == epochs - 1)
                {
                    double validationLoss = ValidateModel(datasetService);
                    Console.WriteLine($"[Época {epoch + 1}] Perda Média de Validação: {validationLoss:F4}");
                }
                
                // Estimativa de tempo restante
                if (epoch < epochs - 1)
                {
                    var avgEpochTime = TimeSpan.FromMilliseconds(totalElapsedTime.TotalMilliseconds / (epoch + 1));
                    var estimatedTimeRemaining = TimeSpan.FromMilliseconds(avgEpochTime.TotalMilliseconds * (epochs - epoch - 1));
                    Console.WriteLine($"[Estimativa] Tempo restante: ~{estimatedTimeRemaining:hh\\:mm\\:ss}");
                }
            }
        }
    }
    private double ValidateModel(DatasetService datasetService)
    {
        Console.WriteLine("\n[Validação] Iniciando validação do modelo...");
        double totalLoss = 0;
        int batchCount = 0;
        datasetService.ResetValidation();
        Stopwatch validationStopwatch = Stopwatch.StartNew();

        while (true)
        {
            var batch = datasetService.GetNextValidationChunk();
            if (batch == null || batch.Count == 0) break;

            var sequenceInputIndices = batch.Select(p => p.InputIndex).ToArray();
            var sequenceTargets = batch.Select(p => p.Target).ToArray();

            // --- CORREÇÃO APLICADA AQUI ---
            // Chama o novo método público que calcula a perda de forma segura e limpa a memória.
            totalLoss += model.CalculateSequenceLoss(sequenceInputIndices, sequenceTargets);
            batchCount++;
        
            Console.Write($"\r[Validação] Processando lote {batchCount}...");
        }

        validationStopwatch.Stop();
        Console.WriteLine($"\r[Validação] Concluída em {validationStopwatch.Elapsed:mm\\:ss}.");
    
        return batchCount > 0 ? totalLoss / batchCount : double.PositiveInfinity;
    }
}