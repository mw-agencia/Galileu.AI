namespace Galileu.Node.Core;

using System;
using System.Text.Json;

public class NeuralNetworkRNN
{
    private Tensor weightsInputHidden;
    private Tensor weightsHiddenHidden;
    private Tensor biasHidden;
    private Tensor weightsHiddenOutput;
    private Tensor biasOutput;
    private readonly int inputSize;
    private readonly int hiddenSize;
    private readonly int outputSize;
    private double[] hiddenState;

    public int InputSize => inputSize;
    public int HiddenSize => hiddenSize;
    public int OutputSize => outputSize;

    // Public getters for weights and biases
    public Tensor WeightsInputHidden => weightsInputHidden;
    public Tensor WeightsHiddenHidden => weightsHiddenHidden;
    public Tensor BiasHidden => biasHidden;
    public Tensor WeightsHiddenOutput => weightsHiddenOutput;
    public Tensor BiasOutput => biasOutput;

    public NeuralNetworkRNN(int inputSize, int hiddenSize, int outputSize)
    {
        this.inputSize = inputSize;
        this.hiddenSize = hiddenSize;
        this.outputSize = outputSize;
        hiddenState = new double[hiddenSize];

        Random rand = new Random();
        double[] weightsInputHiddenData = new double[inputSize * hiddenSize];
        double[] weightsHiddenHiddenData = new double[hiddenSize * hiddenSize];
        double[] biasHiddenData = new double[hiddenSize];
        double[] weightsOutputData = new double[hiddenSize * outputSize];
        double[] biasOutputData = new double[outputSize];

        for (int i = 0; i < weightsInputHiddenData.Length; i++)
            weightsInputHiddenData[i] = rand.NextDouble() - 0.5;
        for (int i = 0; i < weightsHiddenHiddenData.Length; i++)
            weightsHiddenHiddenData[i] = rand.NextDouble() - 0.5;
        for (int i = 0; i < biasHiddenData.Length; i++)
            biasHiddenData[i] = rand.NextDouble() - 0.5;
        for (int i = 0; i < weightsOutputData.Length; i++)
            weightsOutputData[i] = rand.NextDouble() - 0.5;
        for (int i = 0; i < biasOutputData.Length; i++)
            biasOutputData[i] = rand.NextDouble() - 0.5;

        weightsInputHidden = new Tensor(weightsInputHiddenData, new int[] { inputSize, hiddenSize });
        weightsHiddenHidden = new Tensor(weightsHiddenHiddenData, new int[] { hiddenSize, hiddenSize });
        biasHidden = new Tensor(biasHiddenData, new int[] { hiddenSize });
        weightsHiddenOutput = new Tensor(weightsOutputData, new int[] { hiddenSize, outputSize });
        biasOutput = new Tensor(biasOutputData, new int[] { outputSize });
    }

    protected NeuralNetworkRNN(int inputSize, int hiddenSize, int outputSize,
                              Tensor weightsInputHidden, Tensor weightsHiddenHidden, Tensor biasHidden,
                              Tensor weightsHiddenOutput, Tensor biasOutput)
    {
        this.inputSize = inputSize;
        this.hiddenSize = hiddenSize;
        this.outputSize = outputSize;
        this.hiddenState = new double[hiddenSize];
        this.weightsInputHidden = weightsInputHidden ?? throw new ArgumentNullException(nameof(weightsInputHidden));
        this.weightsHiddenHidden = weightsHiddenHidden ?? throw new ArgumentNullException(nameof(weightsHiddenHidden));
        this.biasHidden = biasHidden ?? throw new ArgumentNullException(nameof(biasHidden));
        this.weightsHiddenOutput = weightsHiddenOutput ?? throw new ArgumentNullException(nameof(weightsHiddenOutput));
        this.biasOutput = biasOutput ?? throw new ArgumentNullException(nameof(biasOutput));
    }

    public Tensor Forward(Tensor input)
    {
        if (input == null || input.shape.Length != 1 || input.shape[0] != inputSize)
        {
            throw new ArgumentException("Entrada deve ser unidimensional com tamanho inputSize.");
        }

        double[] newHiddenData = new double[hiddenSize];
        for (int h = 0; h < hiddenSize; h++)
        {
            double sum = 0;
            for (int i = 0; i < inputSize; i++)
            {
                sum += input.Infer(new int[] { i }) * weightsInputHidden.Infer(new int[] { i, h });
            }
            for (int prevH = 0; prevH < hiddenSize; prevH++)
            {
                sum += hiddenState[prevH] * weightsHiddenHidden.Infer(new int[] { prevH, h });
            }
            sum += biasHidden.Infer(new int[] { h });
            newHiddenData[h] = Math.Max(0, sum); // ReLU
        }

        hiddenState = newHiddenData;

        double[] outputData = new double[outputSize];
        double sumExp = 0;
        for (int o = 0; o < outputSize; o++)
        {
            double sum = 0;
            for (int h = 0; h < hiddenSize; h++)
            {
                sum += hiddenState[h] * weightsHiddenOutput.Infer(new int[] { h, o });
            }
            sum += biasOutput.Infer(new int[] { o });
            outputData[o] = Math.Exp(sum);
            sumExp += outputData[o];
        }

        for (int o = 0; o < outputSize; o++)
        {
            outputData[o] /= sumExp;
        }

        return new Tensor(outputData, new int[] { outputSize });
    }

    public double TrainEpoch(Tensor[] inputs, Tensor[] targets, double learningRate)
    {
        double epochLoss = 0;

        for (int i = 0; i < inputs.Length; i++)
        {
            hiddenState = new double[hiddenSize];
            Tensor output = Forward(inputs[i]);

            for (int o = 0; o < outputSize; o++)
            {
                if (targets[i].Infer(new int[] { o }) == 1.0)
                {
                    epochLoss += -Math.Log(output.Infer(new int[] { o }) + 1e-9);
                    break;
                }
            }

            double[] gradOutput = new double[outputSize];
            for (int o = 0; o < outputSize; o++)
            {
                gradOutput[o] = output.Infer(new int[] { o }) - targets[i].Infer(new int[] { o });
            }

            double[] newWeightsOutputData = new double[hiddenSize * outputSize];
            double[] newBiasOutputData = new double[outputSize];
            for (int o = 0; o < outputSize; o++)
            {
                for (int h = 0; h < hiddenSize; h++)
                {
                    int idx = h * outputSize + o;
                    newWeightsOutputData[idx] = weightsHiddenOutput.Infer(new int[] { h, o }) -
                                               learningRate * gradOutput[o] * hiddenState[h];
                }
                newBiasOutputData[o] = biasOutput.Infer(new int[] { o }) - learningRate * gradOutput[o];
            }
            weightsHiddenOutput = new Tensor(newWeightsOutputData, new int[] { hiddenSize, outputSize });
            biasOutput = new Tensor(newBiasOutputData, new int[] { outputSize });

            double[] gradHidden = new double[hiddenSize];
            for (int h = 0; h < hiddenSize; h++)
            {
                double sum = 0;
                for (int o = 0; o < outputSize; o++)
                {
                    sum += gradOutput[o] * weightsHiddenOutput.Infer(new int[] { h, o });
                }
                gradHidden[h] = sum * (hiddenState[h] > 0 ? 1 : 0);
            }

            double[] newWeightsInputHiddenData = new double[inputSize * hiddenSize];
            double[] newWeightsHiddenHiddenData = new double[hiddenSize * hiddenSize];
            double[] newBiasHiddenData = new double[hiddenSize];
            for (int h = 0; h < hiddenSize; h++)
            {
                for (int j = 0; j < inputSize; j++)
                {
                    int idx = j * hiddenSize + h;
                    newWeightsInputHiddenData[idx] = weightsInputHidden.Infer(new int[] { j, h }) -
                                                    learningRate * gradHidden[h] * inputs[i].Infer(new int[] { j });
                }
                for (int prevH = 0; prevH < hiddenSize; prevH++)
                {
                    int idx = prevH * hiddenSize + h;
                    newWeightsHiddenHiddenData[idx] = weightsHiddenHidden.Infer(new int[] { prevH, h }) -
                                                     learningRate * gradHidden[h] * hiddenState[prevH];
                }
                newBiasHiddenData[h] = biasHidden.Infer(new int[] { h }) - learningRate * gradHidden[h];
            }
            weightsInputHidden = new Tensor(newWeightsInputHiddenData, new int[] { inputSize, hiddenSize });
            weightsHiddenHidden = new Tensor(newWeightsHiddenHiddenData, new int[] { hiddenSize, hiddenSize });
            biasHidden = new Tensor(newBiasHiddenData, new int[] { hiddenSize });
        }
        return epochLoss / inputs.Length;
    }

    public void ResetHiddenState()
    {
        hiddenState = new double[hiddenSize];
    }

    public void SaveModel(string filePath)
    {
        try
        {
            var modelData = new NeuralNetworkModelDataRNN
            {
                InputSize = inputSize,
                HiddenSize = hiddenSize,
                OutputSize = outputSize,
                WeightsInputHidden = new TensorData { data = weightsInputHidden.GetData(), shape = weightsInputHidden.GetShape() },
                WeightsHiddenHidden = new TensorData { data = weightsHiddenHidden.GetData(), shape = weightsHiddenHidden.GetShape() },
                BiasHidden = new TensorData { data = biasHidden.GetData(), shape = biasHidden.GetShape() },
                WeightsOutput = new TensorData { data = weightsHiddenOutput.GetData(), shape = weightsHiddenOutput.GetShape() },
                BiasOutput = new TensorData { data = biasOutput.GetData(), shape = biasOutput.GetShape() }
            };

            var options = new JsonSerializerOptions { WriteIndented = true };
            string jsonString = JsonSerializer.Serialize(modelData, options);

            File.WriteAllText(filePath, jsonString);
            Console.WriteLine($"Modelo RNN salvo em JSON em: {filePath}");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Erro ao salvar o modelo RNN: {ex.Message}");
        }
    }

    public static NeuralNetworkRNN LoadModel(string filePath)
    {
        try
        {
            if (!File.Exists(filePath))
            {
                Console.WriteLine($"Arquivo do modelo RNN não encontrado em: {filePath}");
                return null;
            }

            string jsonString = File.ReadAllText(filePath);
            var modelData = JsonSerializer.Deserialize<NeuralNetworkModelDataRNN>(jsonString);

            if (modelData == null)
            {
                throw new Exception("Falha ao desserializar dados do modelo RNN.");
            }

            Tensor loadedWeightsInputHidden = new Tensor(modelData.WeightsInputHidden.data, modelData.WeightsInputHidden.shape);
            Tensor loadedWeightsHiddenHidden = new Tensor(modelData.WeightsHiddenHidden.data, modelData.WeightsHiddenHidden.shape);
            Tensor loadedBiasHidden = new Tensor(modelData.BiasHidden.data, modelData.BiasHidden.shape);
            Tensor loadedWeightsOutput = new Tensor(modelData.WeightsOutput.data, modelData.WeightsOutput.shape);
            Tensor loadedBiasOutput = new Tensor(modelData.BiasOutput.data, modelData.BiasOutput.shape);

            if (loadedWeightsInputHidden.GetShape()[0] != modelData.InputSize || loadedWeightsInputHidden.GetShape()[1] != modelData.HiddenSize)
                throw new Exception("Dimensões de weightsInputHidden não correspondem.");
            if (loadedWeightsHiddenHidden.GetShape()[0] != modelData.HiddenSize || loadedWeightsHiddenHidden.GetShape()[1] != modelData.HiddenSize)
                throw new Exception("Dimensões de weightsHiddenHidden não correspondem.");
            if (loadedBiasHidden.GetShape()[0] != modelData.HiddenSize)
                throw new Exception("Dimensões de biasHidden não correspondem.");
            if (loadedWeightsOutput.GetShape()[0] != modelData.HiddenSize || loadedWeightsOutput.GetShape()[1] != modelData.OutputSize)
                throw new Exception("Dimensões de weightsOutput não correspondem.");
            if (loadedBiasOutput.GetShape()[0] != modelData.OutputSize)
                throw new Exception("Dimensões de biasOutput não correspondem.");

            return new NeuralNetworkRNN(modelData.InputSize, modelData.HiddenSize, modelData.OutputSize,
                                       loadedWeightsInputHidden, loadedWeightsHiddenHidden, loadedBiasHidden,
                                       loadedWeightsOutput, loadedBiasOutput);
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Erro ao carregar o modelo RNN: {ex.Message}");
            return null;
        }
    }
}