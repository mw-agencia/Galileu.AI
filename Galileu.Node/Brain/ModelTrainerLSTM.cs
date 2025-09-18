// --- START OF FILE ModelTrainerLSTM.cs ---

using Galileu.Node.Core;
using Galileu.Node.Services;

namespace Galileu.Node.Brain;

public class ModelTrainerLSTM
{
    private readonly GenerativeNeuralNetworkLSTM model;
    
    public ModelTrainerLSTM(GenerativeNeuralNetworkLSTM model)
    {
        this.model = model ?? throw new ArgumentNullException(nameof(model));
    }

    public void TrainModel(string datasetPath, double learningRate, int epochs, int batchSize, int contextWindowSize, double validationSplit = 0.2)
    {
        // ... (verificações iniciais) ...

        string datasetText = File.ReadAllText(datasetPath);
        var swapFilePath = Path.Combine(Environment.CurrentDirectory, "Dayson", "memory.bin");
        
        using (var datasetService = new DatasetService(swapFilePath))
        {
            // O serviço agora é inicializado com o texto completo
            datasetService.InitializeAndSplit(datasetText, contextWindowSize, model.VocabularyManager.Vocab, "<PAD>", batchSize, validationSplit);

            for (int epoch = 0; epoch < epochs; epoch++)
            {
                double totalEpochLoss = 0;
                int sequenceCount = 0;

                datasetService.ResetTrain(); // Reseta o iterador de treino
                while (true)
                {
                    var batch = datasetService.GetNextTrainChunk();
                    if (batch.Count == 0) break;

                    foreach (var (input, target) in batch)
                    {
                        model.ResetHiddenState();
                        totalEpochLoss += model.TrainEpoch(new[] { input }, new[] { target }, learningRate);
                        sequenceCount++;
                    }
                    Console.Write($"\rÉpoca {epoch + 1}/{epochs}, Processando... {sequenceCount} amostras treinadas.");
                }

                double avgLoss = sequenceCount > 0 ? totalEpochLoss / sequenceCount : double.MaxValue;
                Console.WriteLine($"\nÉpoca {epoch + 1}/{epochs} concluída. Perda de Treino: {avgLoss:F4}");

                double validationLoss = ValidateModel(datasetService);
                Console.WriteLine($"Perda de Validação: {validationLoss:F4}");
            }
        } // O using garante que o Dispose do datasetService seja chamado, limpando o arquivo de swap.
    }

    private double ValidateModel(DatasetService datasetService)
    {
        double totalLoss = 0;
        int count = 0;
        
        datasetService.ResetValidation(); // Reseta o iterador de validação
        while (true)
        {
            var batch = datasetService.GetNextValidationChunk();
            if (batch.Count == 0) break;

            foreach (var (input, target) in batch)
            {
                model.ResetHiddenState();
                Tensor output = model.Forward(input);
                
                // ... (cálculo da perda)
                double loss = 0;
                var outputData = output.GetData();
                var targetData = target.GetData();
                for (int i = 0; i < output.GetTotalSize(); i++)
                {
                    if (targetData[i] == 1.0)
                    {
                        loss = -Math.Log(outputData[i] + 1e-9);
                        break;
                    }
                }
                totalLoss += loss;
                count++;
            }
        }
        return count > 0 ? totalLoss / count : double.MaxValue;
    }
}