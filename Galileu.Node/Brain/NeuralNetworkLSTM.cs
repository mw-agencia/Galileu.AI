using System.Text.Json;
using Galileu.Node.Core;

namespace Galileu.Node.Brain;

public class NeuralNetworkLSTM
{
    private Tensor weightsInputForget; // Pesos para forget gate
    private Tensor weightsHiddenForget;
    private Tensor weightsInputInput; // Pesos para input gate
    private Tensor weightsHiddenInput;
    private Tensor weightsInputCell; // Pesos para cell gate
    private Tensor weightsHiddenCell;
    private Tensor weightsInputOutput; // Pesos para output gate
    private Tensor weightsHiddenOutput;
    private Tensor biasForget;
    private Tensor biasInput;
    private Tensor biasCell;
    private Tensor biasOutput;
    private Tensor weightsHiddenOutputFinal; // Pesos da camada de saída
    private Tensor biasOutputFinal;
    private readonly int inputSize;
    private readonly int hiddenSize;
    private readonly int outputSize;
    private double[] hiddenState;
    private double[] cellState;

    public int InputSize => inputSize;
    public int HiddenSize => hiddenSize;
    public int OutputSize => outputSize;

    // Propriedades públicas para acesso em ModelSerializerLSTM
    public Tensor WeightsInputForget => weightsInputForget;
    public Tensor WeightsHiddenForget => weightsHiddenForget;
    public Tensor WeightsInputInput => weightsInputInput;
    public Tensor WeightsHiddenInput => weightsHiddenInput;
    public Tensor WeightsInputCell => weightsInputCell;
    public Tensor WeightsHiddenCell => weightsHiddenCell;
    public Tensor WeightsInputOutput => weightsInputOutput;
    public Tensor WeightsHiddenOutput => weightsHiddenOutput;
    public Tensor BiasForget => biasForget;
    public Tensor BiasInput => biasInput;
    public Tensor BiasCell => biasCell;
    public Tensor BiasOutput => biasOutput;
    public Tensor WeightsHiddenOutputFinal => weightsHiddenOutputFinal;
    public Tensor BiasOutputFinal => biasOutputFinal;

    public NeuralNetworkLSTM(int inputSize, int hiddenSize, int outputSize)
    {
        this.inputSize = inputSize;
        this.hiddenSize = hiddenSize;
        this.outputSize = outputSize;
        hiddenState = new double[hiddenSize];
        cellState = new double[hiddenSize];

        Random rand = new Random();
        weightsInputForget = InitializeTensor(inputSize, hiddenSize, rand);
        weightsHiddenForget = InitializeTensor(hiddenSize, hiddenSize, rand);
        weightsInputInput = InitializeTensor(inputSize, hiddenSize, rand);
        weightsHiddenInput = InitializeTensor(hiddenSize, hiddenSize, rand);
        weightsInputCell = InitializeTensor(inputSize, hiddenSize, rand);
        weightsHiddenCell = InitializeTensor(hiddenSize, hiddenSize, rand);
        weightsInputOutput = InitializeTensor(inputSize, hiddenSize, rand);
        weightsHiddenOutput = InitializeTensor(hiddenSize, hiddenSize, rand);
        biasForget = InitializeTensor(hiddenSize, rand);
        biasInput = InitializeTensor(hiddenSize, rand);
        biasCell = InitializeTensor(hiddenSize, rand);
        biasOutput = InitializeTensor(hiddenSize, rand);
        weightsHiddenOutputFinal = InitializeTensor(hiddenSize, outputSize, rand);
        biasOutputFinal = InitializeTensor(outputSize, rand);
    }

    protected NeuralNetworkLSTM(int inputSize, int hiddenSize, int outputSize,
                               Tensor weightsInputForget, Tensor weightsHiddenForget,
                               Tensor weightsInputInput, Tensor weightsHiddenInput,
                               Tensor weightsInputCell, Tensor weightsHiddenCell,
                               Tensor weightsInputOutput, Tensor weightsHiddenOutput,
                               Tensor biasForget, Tensor biasInput, Tensor biasCell, Tensor biasOutput,
                               Tensor weightsHiddenOutputFinal, Tensor biasOutputFinal)
    {
        this.inputSize = inputSize;
        this.hiddenSize = hiddenSize;
        this.outputSize = outputSize;
        this.hiddenState = new double[hiddenSize];
        this.cellState = new double[hiddenSize];
        this.weightsInputForget = weightsInputForget ?? throw new ArgumentNullException(nameof(weightsInputForget));
        this.weightsHiddenForget = weightsHiddenForget ?? throw new ArgumentNullException(nameof(weightsHiddenForget));
        this.weightsInputInput = weightsInputInput ?? throw new ArgumentNullException(nameof(weightsInputInput));
        this.weightsHiddenInput = weightsHiddenInput ?? throw new ArgumentNullException(nameof(weightsHiddenInput));
        this.weightsInputCell = weightsInputCell ?? throw new ArgumentNullException(nameof(weightsInputCell));
        this.weightsHiddenCell = weightsHiddenCell ?? throw new ArgumentNullException(nameof(weightsHiddenCell));
        this.weightsInputOutput = weightsInputOutput ?? throw new ArgumentNullException(nameof(weightsInputOutput));
        this.weightsHiddenOutput = weightsHiddenOutput ?? throw new ArgumentNullException(nameof(weightsHiddenOutput));
        this.biasForget = biasForget ?? throw new ArgumentNullException(nameof(biasForget));
        this.biasInput = biasInput ?? throw new ArgumentNullException(nameof(biasInput));
        this.biasCell = biasCell ?? throw new ArgumentNullException(nameof(biasCell));
        this.biasOutput = biasOutput ?? throw new ArgumentNullException(nameof(biasOutput));
        this.weightsHiddenOutputFinal = weightsHiddenOutputFinal ?? throw new ArgumentNullException(nameof(weightsHiddenOutputFinal));
        this.biasOutputFinal = biasOutputFinal ?? throw new ArgumentNullException(nameof(biasOutputFinal));
    }

    private Tensor InitializeTensor(int rows, int cols, Random rand)
    {
        double[] data = new double[rows * cols];
        for (int i = 0; i < data.Length; i++)
            data[i] = rand.NextDouble() - 0.5;
        return new Tensor(data, new int[] { rows, cols });
    }

    private Tensor InitializeTensor(int size, Random rand)
    {
        double[] data = new double[size];
        for (int i = 0; i < data.Length; i++)
            data[i] = rand.NextDouble() - 0.5;
        return new Tensor(data, new int[] { size });
    }

    public Tensor Forward(Tensor input)
    {
        if (input == null || input.shape.Length != 1 || input.shape[0] != inputSize)
        {
            throw new ArgumentException("Entrada deve ser unidimensional com tamanho inputSize.");
        }

        double[] newHiddenState = new double[hiddenSize];
        double[] newCellState = new double[hiddenSize];

        for (int h = 0; h < hiddenSize; h++)
        {
            // Forget gate
            double f_t = Sigmoid(ComputeGate(input, weightsInputForget, hiddenState, weightsHiddenForget, biasForget, h));
            // Input gate
            double i_t = Sigmoid(ComputeGate(input, weightsInputInput, hiddenState, weightsHiddenInput, biasInput, h));
            // Cell gate (candidate)
            double C_tilde = Math.Tanh(ComputeGate(input, weightsInputCell, hiddenState, weightsHiddenCell, biasCell, h));
            // Output gate
            double o_t = Sigmoid(ComputeGate(input, weightsInputOutput, hiddenState, weightsHiddenOutput, biasOutput, h));

            // Atualiza cell state: C_t = f_t * C_{t-1} + i_t * C_tilde
            newCellState[h] = f_t * cellState[h] + i_t * C_tilde;
            // Atualiza hidden state: h_t = o_t * tanh(C_t)
            newHiddenState[h] = o_t * Math.Tanh(newCellState[h]);
        }

        hiddenState = newHiddenState;
        cellState = newCellState;

        // Camada de saída
        double[] outputData = new double[outputSize];
        double sumExp = 0;
        for (int o = 0; o < outputSize; o++)
        {
            double sum = 0;
            for (int h = 0; h < hiddenSize; h++)
            {
                sum += hiddenState[h] * weightsHiddenOutputFinal.Infer(new int[] { h, o });
            }
            sum += biasOutputFinal.Infer(new int[] { o });
            outputData[o] = Math.Exp(sum);
            sumExp += outputData[o];
        }

        for (int o = 0; o < outputSize; o++)
        {
            outputData[o] /= sumExp;
        }

        return new Tensor(outputData, new int[] { outputSize });
    }

    private double Sigmoid(double x)
    {
        return 1.0 / (1.0 + Math.Exp(-x));
    }

    private double ComputeGate(Tensor input, Tensor weightsInput, double[] prevHidden, Tensor weightsHidden, Tensor bias, int h)
    {
        double sum = 0;
        for (int i = 0; i < inputSize; i++)
        {
            sum += input.Infer(new int[] { i }) * weightsInput.Infer(new int[] { i, h });
        }
        for (int prevH = 0; prevH < hiddenSize; prevH++)
        {
            sum += prevHidden[prevH] * weightsHidden.Infer(new int[] { prevH, h });
        }
        sum += bias.Infer(new int[] { h });
        return sum;
    }

    public double TrainEpoch(Tensor[] inputs, Tensor[] targets, double learningRate)
    {
        double epochLoss = 0;

        for (int i = 0; i < inputs.Length; i++)
        {
            hiddenState = new double[hiddenSize];
            cellState = new double[hiddenSize];
            Tensor output = Forward(inputs[i]);

            for (int o = 0; o < outputSize; o++)
            {
                if (targets[i].Infer(new int[] { o }) == 1.0)
                {
                    epochLoss += -Math.Log(output.Infer(new int[] { o }) + 1e-9);
                    break;
                }
            }

            // Backpropagation simplificada (sem BPTT completo)
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
                    newWeightsOutputData[idx] = weightsHiddenOutputFinal.Infer(new int[] { h, o }) -
                                               learningRate * gradOutput[o] * hiddenState[h];
                }
                newBiasOutputData[o] = biasOutputFinal.Infer(new int[] { o }) - learningRate * gradOutput[o];
            }
            weightsHiddenOutputFinal = new Tensor(newWeightsOutputData, new int[] { hiddenSize, outputSize });
            biasOutputFinal = new Tensor(newBiasOutputData, new int[] { outputSize });

            // Gradiente da camada oculta (simplificado)
            double[] gradHidden = new double[hiddenSize];
            for (int h = 0; h < hiddenSize; h++)
            {
                double sum = 0;
                for (int o = 0; o < outputSize; o++)
                {
                    sum += gradOutput[o] * weightsHiddenOutputFinal.Infer(new int[] { h, o });
                }
                gradHidden[h] = sum * (Math.Tanh(cellState[h]) > 0 ? 1 : 0); // Aproximação para derivada
            }

            // Atualiza pesos das portas (simplificado)
            UpdateGateWeights(weightsInputForget, weightsHiddenForget, biasForget, inputs[i], gradHidden, learningRate);
            UpdateGateWeights(weightsInputInput, weightsHiddenInput, biasInput, inputs[i], gradHidden, learningRate);
            UpdateGateWeights(weightsInputCell, weightsHiddenCell, biasCell, inputs[i], gradHidden, learningRate);
            UpdateGateWeights(weightsInputOutput, weightsHiddenOutput, biasOutput, inputs[i], gradHidden, learningRate);
        }
        return epochLoss / inputs.Length;
    }

    private void UpdateGateWeights(Tensor weightsInput, Tensor weightsHidden, Tensor bias, Tensor input, double[] gradHidden, double learningRate)
    {
        double[] newWeightsInputData = new double[inputSize * hiddenSize];
        double[] newWeightsHiddenData = new double[hiddenSize * hiddenSize];
        double[] newBiasData = new double[hiddenSize];

        for (int h = 0; h < hiddenSize; h++)
        {
            for (int j = 0; j < inputSize; j++)
            {
                int idx = j * hiddenSize + h;
                newWeightsInputData[idx] = weightsInput.Infer(new int[] { j, h }) -
                                          learningRate * gradHidden[h] * input.Infer(new int[] { j });
            }
            for (int prevH = 0; prevH < hiddenSize; prevH++)
            {
                int idx = prevH * hiddenSize + h;
                newWeightsHiddenData[idx] = weightsHidden.Infer(new int[] { prevH, h }) -
                                           learningRate * gradHidden[h] * hiddenState[prevH];
            }
            newBiasData[h] = bias.Infer(new int[] { h }) - learningRate * gradHidden[h];
        }

        weightsInput = new Tensor(newWeightsInputData, new int[] { inputSize, hiddenSize });
        weightsHidden = new Tensor(newWeightsHiddenData, new int[] { hiddenSize, hiddenSize });
        bias = new Tensor(newBiasData, new int[] { hiddenSize });
    }

    public void ResetHiddenState()
    {
        hiddenState = new double[hiddenSize];
        cellState = new double[hiddenSize];
    }

    public void SaveModel(string filePath)
    {
        try
        {
            var modelData = new NeuralNetworkModelDataLSTM
            {
                InputSize = inputSize,
                HiddenSize = hiddenSize,
                OutputSize = outputSize,
                WeightsInputForget = new TensorData { data = weightsInputForget.GetData(), shape = weightsInputForget.GetShape() },
                WeightsHiddenForget = new TensorData { data = weightsHiddenForget.GetData(), shape = weightsHiddenForget.GetShape() },
                WeightsInputInput = new TensorData { data = weightsInputInput.GetData(), shape = weightsInputInput.GetShape() },
                WeightsHiddenInput = new TensorData { data = weightsHiddenInput.GetData(), shape = weightsHiddenInput.GetShape() },
                WeightsInputCell = new TensorData { data = weightsInputCell.GetData(), shape = weightsInputCell.GetShape() },
                WeightsHiddenCell = new TensorData { data = weightsHiddenCell.GetData(), shape = weightsHiddenCell.GetShape() },
                WeightsInputOutput = new TensorData { data = weightsInputOutput.GetData(), shape = weightsInputOutput.GetShape() },
                WeightsHiddenOutput = new TensorData { data = weightsHiddenOutput.GetData(), shape = weightsHiddenOutput.GetShape() },
                BiasForget = new TensorData { data = biasForget.GetData(), shape = biasForget.GetShape() },
                BiasInput = new TensorData { data = biasInput.GetData(), shape = biasInput.GetShape() },
                BiasCell = new TensorData { data = biasCell.GetData(), shape = biasCell.GetShape() },
                BiasOutput = new TensorData { data = biasOutput.GetData(), shape = biasOutput.GetShape() },
                WeightsHiddenOutputFinal = new TensorData { data = weightsHiddenOutputFinal.GetData(), shape = weightsHiddenOutputFinal.GetShape() },
                BiasOutputFinal = new TensorData { data = biasOutputFinal.GetData(), shape = biasOutputFinal.GetShape() }
            };

            var options = new JsonSerializerOptions { WriteIndented = true };
            string jsonString = JsonSerializer.Serialize(modelData, options);

            File.WriteAllText(filePath, jsonString);
            Console.WriteLine($"Modelo LSTM salvo em JSON em: {filePath}");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Erro ao salvar o modelo LSTM: {ex.Message}");
        }
    }

    public static NeuralNetworkLSTM LoadModel(string filePath)
    {
        try
        {
            if (!File.Exists(filePath))
            {
                Console.WriteLine($"Arquivo do modelo LSTM não encontrado em: {filePath}");
                return null;
            }

            string jsonString = File.ReadAllText(filePath);
            var modelData = JsonSerializer.Deserialize<NeuralNetworkModelDataLSTM>(jsonString);

            if (modelData == null)
            {
                throw new Exception("Falha ao desserializar dados do modelo LSTM.");
            }

            Tensor loadedWeightsInputForget = new Tensor(modelData.WeightsInputForget.data, modelData.WeightsInputForget.shape);
            Tensor loadedWeightsHiddenForget = new Tensor(modelData.WeightsHiddenForget.data, modelData.WeightsHiddenForget.shape);
            Tensor loadedWeightsInputInput = new Tensor(modelData.WeightsInputInput.data, modelData.WeightsInputInput.shape);
            Tensor loadedWeightsHiddenInput = new Tensor(modelData.WeightsHiddenInput.data, modelData.WeightsHiddenInput.shape);
            Tensor loadedWeightsInputCell = new Tensor(modelData.WeightsInputCell.data, modelData.WeightsInputCell.shape);
            Tensor loadedWeightsHiddenCell = new Tensor(modelData.WeightsHiddenCell.data, modelData.WeightsHiddenCell.shape);
            Tensor loadedWeightsInputOutput = new Tensor(modelData.WeightsInputOutput.data, modelData.WeightsInputOutput.shape);
            Tensor loadedWeightsHiddenOutput = new Tensor(modelData.WeightsHiddenOutput.data, modelData.WeightsHiddenOutput.shape);
            Tensor loadedBiasForget = new Tensor(modelData.BiasForget.data, modelData.BiasForget.shape);
            Tensor loadedBiasInput = new Tensor(modelData.BiasInput.data, modelData.BiasInput.shape);
            Tensor loadedBiasCell = new Tensor(modelData.BiasCell.data, modelData.BiasCell.shape);
            Tensor loadedBiasOutput = new Tensor(modelData.BiasOutput.data, modelData.BiasOutput.shape);
            Tensor loadedWeightsHiddenOutputFinal = new Tensor(modelData.WeightsHiddenOutputFinal.data, modelData.WeightsHiddenOutputFinal.shape);
            Tensor loadedBiasOutputFinal = new Tensor(modelData.BiasOutputFinal.data, modelData.BiasOutputFinal.shape);

            // Validações de dimensões
            if (loadedWeightsInputForget.GetShape()[0] != modelData.InputSize || loadedWeightsInputForget.GetShape()[1] != modelData.HiddenSize)
                throw new Exception("Dimensões de weightsInputForget não correspondem.");
            // (Validações semelhantes para outros tensores)

            return new NeuralNetworkLSTM(modelData.InputSize, modelData.HiddenSize, modelData.OutputSize,
                                        loadedWeightsInputForget, loadedWeightsHiddenForget,
                                        loadedWeightsInputInput, loadedWeightsHiddenInput,
                                        loadedWeightsInputCell, loadedWeightsHiddenCell,
                                        loadedWeightsInputOutput, loadedWeightsHiddenOutput,
                                        loadedBiasForget, loadedBiasInput, loadedBiasCell, loadedBiasOutput,
                                        loadedWeightsHiddenOutputFinal, loadedBiasOutputFinal);
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Erro ao carregar o modelo LSTM: {ex.Message}");
            return null;
        }
    }
}