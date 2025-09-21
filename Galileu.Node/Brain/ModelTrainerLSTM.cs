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

    public void TrainModel(string datasetPath, double learningRate, int epochs, int batchSize, int contextWindowSize,
        double validationSplit = 0.2)
    {
        if (!File.Exists(datasetPath))
            throw new FileNotFoundException("Arquivo de dataset não encontrado.", datasetPath);

        string datasetText = File.ReadAllText(datasetPath);
        if (string.IsNullOrWhiteSpace(datasetText))
            throw new InvalidOperationException("O arquivo de dataset está vazio.");

        var swapFilePath = Path.Combine(Environment.CurrentDirectory, "Dayson", "memory.bin");

        using (var datasetService = new DatasetService(swapFilePath))
        {
            // O serviço é inicializado com o texto completo
            datasetService.InitializeAndSplit(datasetText, contextWindowSize, model.VocabularyManager.Vocab, "<PAD>",
                batchSize, validationSplit);

            for (int epoch = 0; epoch < epochs; epoch++)
            {
                Console.WriteLine($"\n--- Iniciando Época {epoch + 1}/{epochs} ---");
                double totalEpochLoss = 0;
                int sequenceCount = 0;

                datasetService.ResetTrain(); // Reseta o iterador de treino

                // CORREÇÃO: Loop mais robusto
                while (true)
                {
                    var batch = datasetService.GetNextTrainChunk();
                    if (batch == null || batch.Count == 0)
                    {
                        break; // Fim dos dados de treino
                    }

                    foreach (var (input, target) in batch)
                    {
                        // Adicionando uma verificação para garantir que os dados não são nulos
                        if (input == null || target == null) continue;

                        totalEpochLoss += model.TrainEpoch(new[] { input }, new[] { target }, learningRate);
                        sequenceCount++;
                    }

                    Console.Write($"\rÉpoca {epoch + 1}/{epochs}, Processando... {sequenceCount} amostras treinadas.");
                }

                double avgLoss = sequenceCount > 0 ? totalEpochLoss / sequenceCount : double.PositiveInfinity;
                Console.WriteLine($"\nÉpoca {epoch + 1}/{epochs} concluída. Perda Média de Treino: {avgLoss:F4}");

                // Validação (se houver dados)
                double validationLoss = ValidateModel(datasetService);
                if (validationLoss != double.PositiveInfinity)
                {
                    Console.WriteLine($"Perda Média de Validação: {validationLoss:F4}");
                }
            }
        } // O using garante que o Dispose do datasetService seja chamado, limpando o arquivo de swap.
    }

    private double ValidateModel(DatasetService datasetService)
    {
        double totalLoss = 0;
        int count = 0;

        datasetService.ResetValidation();
        while (true)
        {
            var batch = datasetService.GetNextValidationChunk();
            if (batch == null || batch.Count == 0) break;

            foreach (var (input, target) in batch)
            {
                if (input == null || target == null) continue;

                model.ResetHiddenState();
                Tensor output = model.Forward(input);

                double loss = 0;
                var outputData = output.GetData();
                var targetData = target.GetData();

                // Encontra o índice do alvo (one-hot)
                int targetIndex = -1;
                for (int i = 0; i < target.GetTotalSize(); i++)
                {
                    if (targetData[i] == 1.0)
                    {
                        targetIndex = i;
                        break;
                    }
                }

                if (targetIndex != -1)
                {
                    // Evita Log(0)
                    loss = -Math.Log(outputData[targetIndex] + 1e-9);
                    totalLoss += loss;
                    count++;
                }
            }
        }

        return count > 0 ? totalLoss / count : double.PositiveInfinity;
    }
}