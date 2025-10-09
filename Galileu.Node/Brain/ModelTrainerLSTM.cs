using System;
using System.Diagnostics;
using System.IO;
using System.Linq;
using Galileu.Node.Services;

namespace Galileu.Node.Brain;
public class ModelTrainerLSTM
{
    public readonly GenerativeNeuralNetworkLSTM model;
    private readonly Stopwatch _stopwatch = new Stopwatch();
    private GpuLoadMonitor? _gpuMonitor;
    string logPath = Path.Combine(Environment.CurrentDirectory, "Dayson", "training_log.txt");
    
    private readonly Process _currentProcess;
    private long _peakMemoryUsageMB = 0;

    public ModelTrainerLSTM(GenerativeNeuralNetworkLSTM model)
    {
        this.model = model ?? throw new ArgumentNullException(nameof(model));
        _currentProcess = Process.GetCurrentProcess();
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

        var swapFilePath = Path.Combine(Environment.CurrentDirectory, "Dayson", "memory.bin");
        Console.WriteLine($"[Trainer] Dataset service: {swapFilePath}");

        using (var datasetService = new DatasetService(swapFilePath))
        {
            // --- CORREÇÃO: Passa o batchSize dinâmico para o serviço saber o total de lotes ---
            int currentBatchSize = _gpuMonitor?.CurrentBatchSize ?? batchSize;
            datasetService.InitializeAndSplit(datasetPath, contextWindowSize,
                model.VocabularyManager.Vocab, currentBatchSize, validationSplit);

            Console.WriteLine($"\n[Trainer] Configuração:");
            Console.WriteLine($"  - Épocas: {epochs}");
            Console.WriteLine($"  - Learning Rate: {learningRate}");
            Console.WriteLine($"  - Batch Size Inicial: {currentBatchSize} (adaptativo se GPU)");
            Console.WriteLine($"  - Validação Split: {validationSplit:P0}");
            Console.WriteLine($"  - META: RAM < 2GB constante\n");

            const int PATIENCE = 5;
            double bestValidationLoss = double.PositiveInfinity;
            int epochsWithoutImprovement = 0;
            string bestModelCheckpointPath = Path.Combine(Environment.CurrentDirectory, "Dayson", "best_model.json");

            TimeSpan totalElapsedTime = TimeSpan.Zero;

            for (int epoch = 0; epoch < epochs; epoch++)
            {
                _peakMemoryUsageMB = 0;
                _stopwatch.Restart();
                Console.WriteLine($"\n{'═',80}");
                Console.WriteLine($"ÉPOCA {epoch + 1}/{epochs} >> {DateTime.UtcNow}");
                Console.WriteLine($"{'═',80}");

                // --- TREINAMENTO DA ÉPOCA ---
                datasetService.ResetTrain(); // Garante o reset do índice de treino
                int totalTrainBatches = datasetService.GetTotalTrainBatches();
                Console.WriteLine($"[Treino] Iniciando treino com {totalTrainBatches} lotes...");
                
                double totalEpochLoss = 0;
                int batchCount = 0;
                
                while (true)
                {
                    var batch = datasetService.GetNextTrainChunk();
                    if (batch == null || batch.Count == 0) break;

                    batchCount++;
                    Console.Write($"\rÉpoca: {epoch + 1}/{epochs} | Lotes Processados: {batchCount} ...");
                    var sequenceInputIndices = batch.Select(p => p.InputIndex).ToArray();
                    var sequenceTargetIndices = batch.Select(p => p.TargetIndex).ToArray();

                    model.ResetHiddenState();
                    totalEpochLoss += model.TrainSequence(sequenceInputIndices, sequenceTargetIndices, learningRate);

                    if (batchCount % 10 == 0 || batchCount == totalTrainBatches)
                    {
                        _currentProcess.Refresh();
                        long ram = _currentProcess.WorkingSet64 / (1024 * 1024);
                        if (ram > _peakMemoryUsageMB) _peakMemoryUsageMB = ram;
                        
                        double avgLoss = totalEpochLoss / batchCount;
                    }

                    if (_gpuMonitor != null)
                    {
                        datasetService.SetBatchSize(_gpuMonitor.CurrentBatchSize);
                    }
                }

                _stopwatch.Stop();
                totalElapsedTime += _stopwatch.Elapsed;
                double finalEpochLoss = batchCount > 0 ? totalEpochLoss / batchCount : 0;
                
                var logMessage = $"Época {epoch + 1}/{epochs} concluída. Perda média: {finalEpochLoss:F4}";
                Console.WriteLine(logMessage);
                File.AppendAllText(logPath, logMessage + Environment.NewLine);

                // --- VALIDAÇÃO DA ÉPOCA ---
                double validationLoss = ValidateModel(datasetService);
                var validate = $"[Época {epoch + 1}] Perda Final (Validação): {validationLoss:F4}";
                Console.WriteLine(validate);
                File.AppendAllText(logPath, validate + Environment.NewLine);
                bestModelCheckpointPath = Path.Combine(Environment.CurrentDirectory, "Dayson", $"Dayson_{epoch + 1}.json");
                model.SaveModel(bestModelCheckpointPath);

                if (validationLoss < bestValidationLoss)
                {
                    Console.ForegroundColor = ConsoleColor.Green;
                    Console.WriteLine($"  Melhora na validação! ({bestValidationLoss:F4} -> {validationLoss:F4}). Salvando melhor modelo...");
                    Console.ResetColor();
                    bestValidationLoss = validationLoss;
                    model.SaveModel(bestModelCheckpointPath);
                    epochsWithoutImprovement = 0;
                }
                else
                {
                    epochsWithoutImprovement++;
                    Console.ForegroundColor = ConsoleColor.Yellow;
                    Console.WriteLine($"  Sem melhora na validação por {epochsWithoutImprovement} época(s). Melhor perda: {bestValidationLoss:F4}");
                    Console.ResetColor();
                }

                if (epochsWithoutImprovement >= PATIENCE)
                {
                    Console.ForegroundColor = ConsoleColor.Red;
                    Console.WriteLine($"\n[Early Stopping] Parando o treinamento. A validação não melhora há {PATIENCE} épocas.");
                    Console.WriteLine($"O melhor modelo foi salvo em: {bestModelCheckpointPath}");
                    Console.ResetColor();
                    return;
                }
                
                if (epoch < epochs - 1)
                {
                    var avgEpochTime = TimeSpan.FromMilliseconds(totalElapsedTime.TotalMilliseconds / (epoch + 1));
                    var estimatedTimeRemaining = TimeSpan.FromMilliseconds(avgEpochTime.TotalMilliseconds * (epochs - epoch - 1));
                    Console.WriteLine($"[Estimativa] Tempo restante: ~{estimatedTimeRemaining:hh\\:mm\\:ss}");
                }
            }
            bestModelCheckpointPath = Path.Combine(Environment.CurrentDirectory, "Dayson", "dayson_model.json");
            model.SaveModel(bestModelCheckpointPath);
        }
    }

    private double ValidateModel(DatasetService datasetService)
    {
        datasetService.ResetValidation(); // Garante o reset do índice de validação
        int totalValidationBatches = datasetService.GetTotalValidationBatches();
        Console.WriteLine($"\n[Validação] Iniciando com {totalValidationBatches} lotes...");
        
        double totalLoss = 0;
        int batchCount = 0;
        Stopwatch validationStopwatch = Stopwatch.StartNew();

        while (true)
        {
            var batch = datasetService.GetNextValidationChunk();
            if (batch == null || batch.Count == 0) break;

            batchCount++;
            var sequenceInputIndices = batch.Select(p => p.InputIndex).ToArray();
            var sequenceTargetIndices = batch.Select(p => p.TargetIndex).ToArray();

            totalLoss += model.CalculateSequenceLoss(sequenceInputIndices, sequenceTargetIndices);
        }

        validationStopwatch.Stop();
        Console.WriteLine($"[Validação] Concluída em {validationStopwatch.Elapsed:mm\\:ss}.");
    
        return batchCount > 0 ? totalLoss / batchCount : double.PositiveInfinity;
    }
}