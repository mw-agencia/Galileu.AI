using System.Diagnostics;
using Galileu.Node.Services;

namespace Galileu.Node.Brain;

/// <summary>
/// Orquestra o processo de treinamento de um modelo GenerativeNeuralNetworkLSTM.
/// OTIMIZADO: Gerenciamento agressivo de memória para treinamentos de longo prazo (100+ épocas).
/// </summary>
public class ModelTrainerLSTM
{
    public readonly GenerativeNeuralNetworkLSTM model;
    private readonly Stopwatch _stopwatch = new Stopwatch();
    private GpuLoadMonitor? _gpuMonitor;
    // === NOVO: Monitoramento de memória ===
    private readonly Process _currentProcess;
    private long _peakMemoryUsageMB = 0;
    string logPath = Path.Combine(Environment.CurrentDirectory, "Dayson", "training_log.txt");

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
            Console.WriteLine($"  - Validação Split: {validationSplit:P0}");
            Console.WriteLine($"  - META: RAM < 10GB constante\n");

            TimeSpan totalElapsedTime = TimeSpan.Zero;

            for (int epoch = 0; epoch < epochs; epoch++)
            {
                _stopwatch.Restart();
                Console.WriteLine($"\n{'═',60}");
                Console.WriteLine($"ÉPOCA {epoch + 1}/{epochs} >> Learning Rate: {learningRate} >> {DateTime.UtcNow}");
                Console.WriteLine($"{'═',60}");

                double totalEpochLoss = 0;
                int batchCount = 0;
                datasetService.ResetTrain();
                var batchStopwatch = Stopwatch.StartNew();

                while (true)
                {
                    var batch = datasetService.GetNextTrainChunk();
                    if (batch == null || batch.Count == 0) break;

                    var sequenceInputIndices = batch.Select(p => p.InputIndex).ToArray();
                    var sequenceTargets = batch.Select(p => p.Target).ToArray();

                    model.ResetHiddenState();

                    batchStopwatch.Restart();
                    totalEpochLoss += model.TrainSequence(sequenceInputIndices, sequenceTargets, learningRate);
                    batchStopwatch.Stop();

                    batchCount++;
                    Console.Write($"\rÉpoca: {epoch + 1}/{epochs} | Lotes: {batchCount} ...");
                    model._tensorPool?.Trim();
                    ForceGarbageCollection();
                    
                }

                _stopwatch.Stop();
                totalElapsedTime += _stopwatch.Elapsed;
                double avgLoss = batchCount > 0 ? totalEpochLoss / batchCount : double.PositiveInfinity;
                var logMessage = $"Época {epoch + 1}/{epochs} concluída. Perda média: {avgLoss:F4} : Eponca concluída em {_stopwatch.Elapsed:hh\\\\:mm\\\\:ss}";
                Console.WriteLine(logMessage);
                File.AppendAllText(logPath, logMessage + Environment.NewLine);
                Console.WriteLine($"  RAM Atual: {GetCurrentMemoryUsageMB()}MB");
                Console.WriteLine($"  RAM Pico: {_peakMemoryUsageMB}MB");

                // === NOVO: Limpeza completa entre épocas ===
                Console.WriteLine("\n[Trainer] Limpeza entre épocas...");
                CleanupBetweenEpochs();
                
                double validationLoss = ValidateModel(datasetService);
                var logvalidationLoss = $"[Época {epoch + 1}] Perda Média de Validação: {validationLoss:F4}";
                Console.WriteLine(logMessage);
                File.AppendAllText(logPath, logvalidationLoss + Environment.NewLine);
                
                // Estimativa de tempo restante
                if (epoch < epochs - 1)
                {
                    var avgEpochTime = TimeSpan.FromMilliseconds(totalElapsedTime.TotalMilliseconds / (epoch + 1));
                    var estimatedTimeRemaining = TimeSpan.FromMilliseconds(avgEpochTime.TotalMilliseconds * (epochs - epoch - 1));
                    Console.WriteLine($"[Estimativa] Tempo restante: ~{estimatedTimeRemaining:hh\\:mm\\:ss}");
                }
            }
            
            // === NOVO: Relatório final de memória ===
            Console.WriteLine("\n" + new string('═', 60));
            Console.WriteLine("TREINAMENTO CONCLUÍDO");
            Console.WriteLine(new string('═', 60));
            Console.WriteLine($"RAM Pico durante treinamento: {_peakMemoryUsageMB}MB");
            Console.WriteLine($"RAM Final: {GetCurrentMemoryUsageMB()}MB");
            
            if (_peakMemoryUsageMB > 10000)
            {
                Console.ForegroundColor = ConsoleColor.Red;
                Console.WriteLine($"[ALERTA] Pico de RAM excedeu 10GB: {_peakMemoryUsageMB}MB");
                Console.ResetColor();
            }
            else
            {
                Console.ForegroundColor = ConsoleColor.Green;
                Console.WriteLine($"[SUCESSO] RAM mantida abaixo de 10GB");
                Console.ResetColor();
            }
        }
    }

    /// <summary>
    /// NOVO: Limpeza agressiva de memória entre épocas.
    /// </summary>
    private void CleanupBetweenEpochs()
    {
        long memoryBefore = GetCurrentMemoryUsageMB();
        
        // 1. Limpa TensorPool completamente
        if (model._tensorPool != null)
        {
            Console.WriteLine("  [Cleanup] TensorPool.Trim()");
            model._tensorPool.Trim();
        }
        
        // 2. Reseta hidden states
        Console.WriteLine("  [Cleanup] ResetHiddenState()");
        model.ResetHiddenState();
        
        // 3. Força coleta de lixo completa (Gen 2)
        Console.WriteLine("  [Cleanup] GC.Collect(Gen 2)");
        ForceGarbageCollection();
        
        long memoryAfter = GetCurrentMemoryUsageMB();
        long freed = memoryBefore - memoryAfter;
        
        Console.WriteLine($"  [Cleanup] Liberado: {freed}MB (Antes: {memoryBefore}MB → Depois: {memoryAfter}MB)");
    }
    private void ForceGarbageCollection()
    {
        GC.Collect(2, GCCollectionMode.Forced, true, true);
        GC.WaitForPendingFinalizers();
        GC.Collect(2, GCCollectionMode.Forced, true, true);
    }
    private long GetCurrentMemoryUsageMB()
    {
        _currentProcess.Refresh();
        return _currentProcess.WorkingSet64 / (1024 * 1024);
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

            totalLoss += model.CalculateSequenceLoss(sequenceInputIndices, sequenceTargets);
            batchCount++;
        
            Console.Write($"\r[Validação] Processando lote {batchCount}...");
            
            // === NOVO: Limpeza durante validação para evitar acúmulo ===
            if (batchCount % 20 == 0)
            {
                ForceGarbageCollection();
            }
        }

        validationStopwatch.Stop();
        Console.WriteLine($"\r[Validação] Concluída em {validationStopwatch.Elapsed:mm\\:ss} | RAM: {GetCurrentMemoryUsageMB()}MB");
    
        return batchCount > 0 ? totalLoss / batchCount : double.PositiveInfinity;
    }
}