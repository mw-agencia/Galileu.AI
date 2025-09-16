namespace Galileu.Node.Core;

public class NeuralNetworkModelDataRNN
{
    public int InputSize { get; set; }
    public int HiddenSize { get; set; }
    public int OutputSize { get; set; }
    public TensorData WeightsInputHidden { get; set; }
    public TensorData WeightsHiddenHidden { get; set; }
    public TensorData BiasHidden { get; set; }
    public TensorData WeightsOutput { get; set; }
    public TensorData BiasOutput { get; set; }
}