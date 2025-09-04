using System.Text.Json;

namespace Galileu.Core;

public class NeuralNetwork
    {
        private Tensor weightsHidden;
        private Tensor biasHidden;
        private Tensor weightsOutput;
        private Tensor biasOutput;
        private readonly int inputSize;
        private readonly int hiddenSize;
        private readonly int outputSize;

        public int InputSize => inputSize;
        public int HiddenSize => hiddenSize;
        public int OutputSize => outputSize;

        public NeuralNetwork(int inputSize, int hiddenSize, int outputSize)
        {
            this.inputSize = inputSize;
            this.hiddenSize = hiddenSize;
            this.outputSize = outputSize;

            Random rand = new Random();
            double[] weightsHiddenData = new double[inputSize * hiddenSize];
            double[] biasHiddenData = new double[hiddenSize];
            double[] weightsOutputData = new double[hiddenSize * outputSize];
            double[] biasOutputData = new double[outputSize];

            for (int i = 0; i < weightsHiddenData.Length; i++)
                weightsHiddenData[i] = rand.NextDouble() - 0.5;
            for (int i = 0; i < biasHiddenData.Length; i++)
                biasHiddenData[i] = rand.NextDouble() - 0.5;
            for (int i = 0; i < weightsOutputData.Length; i++)
                weightsOutputData[i] = rand.NextDouble() - 0.5;
            for (int i = 0; i < biasOutputData.Length; i++)
                biasOutputData[i] = rand.NextDouble() - 0.5;

            weightsHidden = new Tensor(weightsHiddenData, new int[] { inputSize, hiddenSize });
            biasHidden = new Tensor(biasHiddenData, new int[] { hiddenSize });
            weightsOutput = new Tensor(weightsOutputData, new int[] { hiddenSize, outputSize });
            biasOutput = new Tensor(biasOutputData, new int[] { outputSize });
        }

        // Construtor privado para ser usado pelo método LoadModel (para reconstruir a rede)
        private NeuralNetwork(int inputSize, int hiddenSize, int outputSize,
                              Tensor weightsHidden, Tensor biasHidden,
                              Tensor weightsOutput, Tensor biasOutput)
        {
            this.inputSize = inputSize;
            this.hiddenSize = hiddenSize;
            this.outputSize = outputSize;
            this.weightsHidden = weightsHidden;
            this.biasHidden = biasHidden;
            this.weightsOutput = weightsOutput;
            this.biasOutput = biasOutput;
        }

        public Tensor Forward(Tensor input)
        {
            if (input.shape.Length != 1 || input.shape[0] != inputSize)
            {
                throw new ArgumentException(
                    "O tensor de entrada deve ser unidimensional com tamanho igual a inputSize.");
            }

            double[] hiddenData = new double[hiddenSize];
            for (int h = 0; h < hiddenSize; h++)
            {
                double sum = 0;
                for (int i = 0; i < inputSize; i++)
                {
                    sum += input.Infer(new int[] { i }) * weightsHidden.Infer(new int[] { i, h });
                }

                sum += biasHidden.Infer(new int[] { h });
                hiddenData[h] = Math.Max(0, sum); // ReLU
            }

            Tensor hidden = new Tensor(hiddenData, new int[] { hiddenSize });

            double[] outputData = new double[outputSize];
            double sumExp = 0;
            for (int o = 0; o < outputSize; o++)
            {
                double sum = 0;
                for (int h = 0; h < hiddenSize; h++)
                {
                    sum += hidden.Infer(new int[] { h }) * weightsOutput.Infer(new int[] { h, o });
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
                Tensor hidden = ComputeHidden(inputs[i]);
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
                        newWeightsOutputData[idx] = weightsOutput.Infer(new int[] { h, o }) -
                                                    learningRate * gradOutput[o] * hidden.Infer(new int[] { h });
                    }
                    newBiasOutputData[o] = biasOutput.Infer(new int[] { o }) -
                                           learningRate * gradOutput[o];
                }

                weightsOutput = new Tensor(newWeightsOutputData, new int[] { hiddenSize, outputSize });
                biasOutput = new Tensor(newBiasOutputData, new int[] { outputSize });


                double[] gradHidden = new double[hiddenSize];
                for (int h = 0; h < hiddenSize; h++)
                {
                    double sum = 0;
                    for (int o = 0; o < outputSize; o++)
                    {
                        sum += gradOutput[o] * weightsOutput.Infer(new int[] { h, o });
                    }
                    gradHidden[h] = sum * (hidden.Infer(new int[] { h }) > 0 ? 1 : 0);
                }

                double[] newWeightsHiddenData = new double[inputSize * hiddenSize];
                double[] newBiasHiddenData = new double[hiddenSize];
                for (int h = 0; h < hiddenSize; h++)
                {
                    for (int j = 0; j < inputSize; j++)
                    {
                        int idx = j * hiddenSize + h;
                        newWeightsHiddenData[idx] = weightsHidden.Infer(new int[] { j, h }) -
                                                    learningRate * gradHidden[h] * inputs[i].Infer(new int[] { j });
                    }
                    newBiasHiddenData[h] = biasHidden.Infer(new int[] { h }) -
                                           learningRate * gradHidden[h];
                }

                weightsHidden = new Tensor(newWeightsHiddenData, new int[] { inputSize, hiddenSize });
                biasHidden = new Tensor(newBiasHiddenData, new int[] { hiddenSize });
            }
            return epochLoss / inputs.Length;
        }

        private Tensor ComputeHidden(Tensor input)
        {
            double[] hiddenData = new double[hiddenSize];
            for (int h = 0; h < hiddenSize; h++)
            {
                double sum = 0;
                for (int i = 0; i < inputSize; i++)
                {
                    sum += input.Infer(new int[] { i }) * weightsHidden.Infer(new int[] { i, h });
                }

                sum += biasHidden.Infer(new int[] { h });
                hiddenData[h] = Math.Max(0, sum); // ReLU
            }

            return new Tensor(hiddenData, new int[] { hiddenSize });
        }

        // NOVO MÉTODO SaveModel - Serializa para JSON usando System.Text.Json
        public void SaveModel(string filePath)
        {
            try
            {
                var modelData = new NeuralNetworkModelData
                {
                    InputSize = inputSize,
                    HiddenSize = hiddenSize,
                    OutputSize = outputSize,
                    WeightsHidden = new TensorData { data = weightsHidden.GetData(), shape = weightsHidden.GetShape() },
                    BiasHidden = new TensorData { data = biasHidden.GetData(), shape = biasHidden.GetShape() },
                    WeightsOutput = new TensorData { data = weightsOutput.GetData(), shape = weightsOutput.GetShape() },
                    BiasOutput = new TensorData { data = biasOutput.GetData(), shape = biasOutput.GetShape() }
                };

                // Opções de serialização para formatação legível (opcional)
                var options = new JsonSerializerOptions { WriteIndented = true };
                string jsonString = JsonSerializer.Serialize(modelData, options);

                File.WriteAllText(filePath, jsonString);
                Console.WriteLine($"Modelo salvo em JSON (System.Text.Json) em: {filePath}");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Erro ao salvar o modelo em JSON (System.Text.Json): {ex.Message}");
            }
        }

        // NOVO MÉTODO LoadModel - Desserializa de JSON usando System.Text.Json
        public static NeuralNetwork LoadModel(string filePath)
        {
            try
            {
                if (!File.Exists(filePath))
                {
                    Console.WriteLine($"Arquivo do modelo JSON não encontrado em: {filePath}");
                    return null;
                }

                string jsonString = File.ReadAllText(filePath);
                
                // Desserializa para a classe DTO
                var modelData = JsonSerializer.Deserialize<NeuralNetworkModelData>(jsonString);

                if (modelData == null)
                {
                    throw new Exception("Falha ao desserializar dados do modelo JSON.");
                }

                // Cria os Tensors a partir dos dados desserializados
                Tensor loadedWeightsHidden = new Tensor(modelData.WeightsHidden.data, modelData.WeightsHidden.shape);
                Tensor loadedBiasHidden = new Tensor(modelData.BiasHidden.data, modelData.BiasHidden.shape);
                Tensor loadedWeightsOutput = new Tensor(modelData.WeightsOutput.data, modelData.WeightsOutput.shape);
                Tensor loadedBiasOutput = new Tensor(modelData.BiasOutput.data, modelData.BiasOutput.shape);

                // Validação das dimensões carregadas
                if (loadedWeightsHidden.GetShape()[0] != modelData.InputSize || loadedWeightsHidden.GetShape()[1] != modelData.HiddenSize)
                    throw new Exception("Dimensões de weightsHidden não correspondem ao modelo carregado.");
                if (loadedBiasHidden.GetShape()[0] != modelData.HiddenSize)
                    throw new Exception("Dimensões de biasHidden não correspondem ao modelo carregado.");
                if (loadedWeightsOutput.GetShape()[0] != modelData.HiddenSize || loadedWeightsOutput.GetShape()[1] != modelData.OutputSize)
                    throw new Exception("Dimensões de weightsOutput não correspondem ao modelo carregado.");
                if (loadedBiasOutput.GetShape()[0] != modelData.OutputSize)
                    throw new Exception("Dimensões de biasOutput não correspondem ao modelo carregado.");

                // Retorna uma nova instância de NeuralNetwork com os dados carregados
                return new NeuralNetwork(modelData.InputSize, modelData.HiddenSize, modelData.OutputSize,
                                         loadedWeightsHidden, loadedBiasHidden,
                                         loadedWeightsOutput, loadedBiasOutput);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Erro ao carregar o modelo JSON (System.Text.Json): {ex.Message}");
                return null;
            }
        }
    }