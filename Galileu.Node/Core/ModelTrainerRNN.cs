namespace Galileu.Node.Core;

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

public class ModelTrainerRNN
{
    private readonly GenerativeNeuralNetworkRNN model;

    public ModelTrainerRNN(GenerativeNeuralNetworkRNN model)
    {
        this.model = model ?? throw new ArgumentNullException(nameof(model));
    }

    public void TrainModel(string datasetPath, double learningRate, int epochs, double validationSplit = 0.2)
    {
        if (!File.Exists(datasetPath))
        {
            throw new FileNotFoundException($"Dataset não encontrado em: {datasetPath}");
        }

        var lines = File.ReadAllLines(datasetPath).Where(l => !string.IsNullOrEmpty(l)).ToList();
        if (lines.Count == 0)
        {
            throw new InvalidOperationException("Dataset está vazio.");
        }

        int validationSize = (int)(lines.Count * validationSplit);
        int trainSize = lines.Count - validationSize;
        if (trainSize == 0)
        {
            throw new InvalidOperationException("Nenhum dado disponível para treinamento após divisão de validação.");
        }

        var trainData = lines.Take(trainSize).ToList();
        var validationData = lines.Skip(trainSize).ToList();

        if (model.VocabularyManager.Vocab.Count == 0)
        {
            throw new InvalidOperationException("Vocabulário vazio. Certifique-se de que o vocabulário foi construído corretamente.");
        }

        List<(Tensor Input, Tensor Target)[]> trainSequences = PrepareSequences(trainData);
        List<(Tensor Input, Tensor Target)[]> validationSequences = PrepareSequences(validationData);

        if (trainSequences.Count == 0)
        {
            throw new InvalidOperationException("Nenhuma sequência válida gerada para treinamento.");
        }

        for (int epoch = 0; epoch < epochs; epoch++)
        {
            double loss = 0;
            int sequenceCount = 0;

            foreach (var sequence in trainSequences)
            {
                if (sequence.Length == 0) continue;

                model.ResetHiddenState();
                foreach (var (input, target) in sequence)
                {
                    try
                    {
                        loss += model.TrainEpoch(new[] { input }, new[] { target }, learningRate);
                        sequenceCount++;
                    }
                    catch (Exception ex)
                    {
                        Console.WriteLine($"Erro ao treinar sequência: {ex.Message}");
                    }
                }
            }

            double avgLoss = sequenceCount > 0 ? loss / sequenceCount : double.MaxValue;
            Console.WriteLine($"Época {epoch + 1}/{epochs}, Perda: {avgLoss:F4}");

            double validationLoss = ValidateModel(validationSequences);
            Console.WriteLine($"Validação, Perda: {validationLoss:F4}");
        }
    }

    private List<(Tensor Input, Tensor Target)[]> PrepareSequences(List<string> data)
    {
        List<(Tensor Input, Tensor Target)[]> sequences = new List<(Tensor Input, Tensor Target)[]>();
        foreach (var line in data)
        {
            var tokens = line.Split(new[] { ' ' }, StringSplitOptions.RemoveEmptyEntries);
            if (tokens.Length < 2) continue;

            List<(Tensor Input, Tensor Target)> seq = new List<(Tensor Input, Tensor Target)>();
            for (int i = 0; i < tokens.Length - 1; i++)
            {
                double[] inputData = new double[model.InputSize];
                double[] targetData = new double[model.OutputSize];

                string token = tokens[i].Trim();
                string nextToken = tokens[i + 1].Trim();

                if (model.VocabularyManager.Vocab.ContainsKey(token))
                {
                    inputData[0] = model.VocabularyManager.Vocab[token];
                }
                else
                {
                    inputData[0] = model.VocabularyManager.Vocab.ContainsKey("<UNK>")
                        ? model.VocabularyManager.Vocab["<UNK>"]
                        : 0;
                }

                if (model.VocabularyManager.Vocab.ContainsKey(nextToken))
                {
                    targetData[model.VocabularyManager.Vocab[nextToken]] = 1.0;
                }
                else
                {
                    targetData[model.VocabularyManager.Vocab.ContainsKey("<UNK>")
                        ? model.VocabularyManager.Vocab["<UNK>"]
                        : 0] = 1.0;
                }

                try
                {
                    seq.Add((new Tensor(inputData, new int[] { model.InputSize }),
                             new Tensor(targetData, new int[] { model.OutputSize })));
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"Erro ao criar tensor para token '{token}': {ex.Message}");
                    continue;
                }
            }

            if (seq.Count > 0)
            {
                sequences.Add(seq.ToArray());
            }
        }
        return sequences;
    }

    private double ValidateModel(List<(Tensor Input, Tensor Target)[]> validationSequences)
    {
        double totalLoss = 0;
        int count = 0;

        foreach (var sequence in validationSequences)
        {
            if (sequence.Length == 0) continue;

            model.ResetHiddenState();
            foreach (var (input, target) in sequence)
            {
                try
                {
                    Tensor output = model.Forward(input);
                    double loss = 0;
                    for (int o = 0; o < model.OutputSize; o++)
                    {
                        if (target.Infer(new int[] { o }) == 1.0)
                        {
                            loss += -Math.Log(output.Infer(new int[] { o }) + 1e-9);
                            break;
                        }
                    }
                    totalLoss += loss;
                    count++;
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"Erro ao validar sequência: {ex.Message}");
                }
            }
        }

        return count > 0 ? totalLoss / count : double.MaxValue;
    }
}