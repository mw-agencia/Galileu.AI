namespace Galileu.Node.Core;

using System;
using System.IO;
using System.Text.Json;

public class ModelSerializerRNN
{
    public static void SaveModel(GenerativeNeuralNetworkRNN model, string filePath)
    {
        if (model == null)
        {
            throw new ArgumentNullException(nameof(model));
        }
        model.SaveModel(filePath);
    }

    public static GenerativeNeuralNetworkRNN LoadModel(string filePath)
    {
        try
        {
            var baseModel = NeuralNetworkRNN.LoadModel(filePath);
            if (baseModel == null)
            {
                Console.WriteLine("Falha ao carregar o modelo base.");
                return null;
            }

            var vocabManager = new VocabularyManager();
            int vocabSize = vocabManager.LoadVocabulary();
            if (vocabSize == 0)
            {
                Console.WriteLine("Vocabulário vazio ou não encontrado.");
                return null;
            }

            return new GenerativeNeuralNetworkRNN(
                baseModel.InputSize,
                baseModel.HiddenSize,
                baseModel.OutputSize,
                baseModel.WeightsInputHidden,
                baseModel.WeightsHiddenHidden,
                baseModel.BiasHidden,
                baseModel.WeightsHiddenOutput,
                baseModel.BiasOutput,
                vocabManager);
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Erro ao carregar o modelo RNN generativo: {ex.Message}");
            return null;
        }
    }
}