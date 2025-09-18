using Galileu.Node.Core;

namespace Galileu.Node.Brain;

public class ModelSerializerLSTM
{
    public static void SaveModel(GenerativeNeuralNetworkLSTM model, string filePath)
    {
        if (model == null)
        {
            throw new ArgumentNullException(nameof(model));
        }
        try
        {
            model.SaveModel(filePath);
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Erro ao salvar o modelo LSTM: {ex.Message}");
            throw;
        }
    }

    public static GenerativeNeuralNetworkLSTM LoadModel(string filePath)
    {
        try
        {
            if (!File.Exists(filePath))
            {
                Console.WriteLine($"Arquivo do modelo LSTM não encontrado em: {filePath}");
                return null;
            }

            var baseModel = NeuralNetworkLSTM.LoadModel(filePath);
            if (baseModel == null)
            {
                Console.WriteLine("Falha ao carregar o modelo base LSTM.");
                return null;
            }

            var vocabManager = new VocabularyManager();
            int vocabSize = vocabManager.LoadVocabulary();
            if (vocabSize == 0)
            {
                Console.WriteLine("Vocabulário vazio ou não encontrado em 'vocab.txt'.");
                return null;
            }

            if (vocabSize != baseModel.OutputSize)
            {
                Console.WriteLine($"Tamanho do vocabulário ({vocabSize}) não corresponde ao OutputSize do modelo ({baseModel.OutputSize}).");
                return null;
            }

            return new GenerativeNeuralNetworkLSTM(
                baseModel.InputSize,
                baseModel.HiddenSize,
                baseModel.OutputSize,
                baseModel.WeightsInputForget,
                baseModel.WeightsHiddenForget,
                baseModel.WeightsInputInput,
                baseModel.WeightsHiddenInput,
                baseModel.WeightsInputCell,
                baseModel.WeightsHiddenCell,
                baseModel.WeightsInputOutput,
                baseModel.WeightsHiddenOutput,
                baseModel.BiasForget,
                baseModel.BiasInput,
                baseModel.BiasCell,
                baseModel.BiasOutput,
                baseModel.WeightsHiddenOutputFinal,
                baseModel.BiasOutputFinal,
                vocabManager);
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Erro ao carregar o modelo LSTM generativo: {ex.Message}");
            return null;
        }
    }
}