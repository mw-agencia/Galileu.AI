using Galileu.Node.Core;
using Galileu.Node.Services;

namespace Galileu.Node.Brain;

public class ModelTrainerLSTM
{
    public readonly GenerativeNeuralNetworkLSTM model;

    public ModelTrainerLSTM(GenerativeNeuralNetworkLSTM model)
    {
        this.model = model ?? throw new ArgumentNullException(nameof(model));
    }

    public void TrainModel(string datasetPath, double learningRate, int epochs, int batchSize, int contextWindowSize,
        double validationSplit)
    {
        if (!File.Exists(datasetPath))
            throw new FileNotFoundException("Arquivo de dataset não encontrado.", datasetPath);

        string datasetText = File.ReadAllText(datasetPath);
        if (string.IsNullOrWhiteSpace(datasetText))
            throw new InvalidOperationException("O arquivo de dataset está vazio.");

        var swapFilePath = Path.Combine(Environment.CurrentDirectory, "Dayson", "memory.bin");
        Console.WriteLine($"{nameof(TrainModel)} >> {swapFilePath} >> {DateTime.UtcNow}");

        using (var datasetService = new DatasetService(swapFilePath))
        {
            datasetService.InitializeAndSplit(datasetText, contextWindowSize, model.VocabularyManager.Vocab, "<PAD>",
                batchSize, validationSplit);

            for (int epoch = 0; epoch < epochs; epoch++)
            {
                Console.WriteLine($"\n--- Iniciando Época {epoch + 1}/{epochs} - learningRate : {learningRate}- validationSplit : {validationSplit} >> {DateTime.UtcNow}");
                double totalEpochLoss = 0;
                int sequenceCount = 0;

                datasetService.ResetTrain();

                while (true)
                {
                    var batch = datasetService.GetNextTrainChunk();
                    if (batch == null || batch.Count == 0) break;

                    // --- CORREÇÃO: Agrupa o lote em sequências e treina ---
                    var sequenceInputs = new List<Tensor>();
                    var sequenceTargets = new List<Tensor>();

                    for (int i = 0; i < batch.Count; i++)
                    {
                        var (input, target) = batch[i];
                        if (input == null || target == null) continue;

                        sequenceInputs.Add(input);
                        sequenceTargets.Add(target);

                        // Quando a sequência atinge o tamanho da janela de contexto, ou é o fim do lote, treine.
                        if (sequenceInputs.Count == contextWindowSize || i == batch.Count - 1)
                        {
                            if (sequenceInputs.Any())
                            {
                                // Chama o novo método que implementa BPTT
                                totalEpochLoss += model.TrainSequence(sequenceInputs.ToArray(), sequenceTargets.ToArray(), learningRate);
                                sequenceCount++;

                                // Limpa para a próxima sequência
                                sequenceInputs.Clear();
                                sequenceTargets.Clear();
                            }
                        }
                    }
                     Console.Write($"\rÉpoca {epoch + 1}/{epochs}, Processando... {sequenceCount} sequências treinadas.");
                }

                double avgLoss = sequenceCount > 0 ? totalEpochLoss / sequenceCount : double.PositiveInfinity;
                Console.WriteLine($"\nÉpoca {epoch + 1}/{epochs} concluída. Perda Média de Treino: {avgLoss:F4}");

                double validationLoss = ValidateModel(datasetService, contextWindowSize);
                if (validationLoss != double.PositiveInfinity)
                {
                    Console.WriteLine($"Perda Média de Validação: {validationLoss:F4}");
                }
            }
        }
    }

    // A validação também deve processar sequências para ser consistente
    private double ValidateModel(DatasetService datasetService, int contextWindowSize)
    {
        double totalLoss = 0;
        int sequenceCount = 0;

        datasetService.ResetValidation();
        while (true)
        {
            var batch = datasetService.GetNextValidationChunk();
            if (batch == null || batch.Count == 0) break;
        
            model.ResetHiddenState();
            foreach (var (input, target) in batch)
            {
                if (input == null || target == null) continue;
            
                // CORREÇÃO: Chame ForwardCpu diretamente para garantir consistência
                // com a lógica de treinamento da CPU e evitar o placeholder da GPU.
                Tensor output = model.Forward(input); // <-- MUDANÇA AQUI
            
                var outputData = output.GetData();
                var targetData = target.GetData();
            
                int targetIndex = Array.IndexOf(targetData, 1.0);
                if (targetIndex != -1)
                {
                    totalLoss += -Math.Log(outputData[targetIndex] + 1e-9);
                }
            }
            sequenceCount += batch.Count;
        }

        return sequenceCount > 0 ? totalLoss / sequenceCount : double.PositiveInfinity;
    }
}