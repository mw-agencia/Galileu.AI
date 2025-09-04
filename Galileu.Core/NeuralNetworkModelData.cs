namespace Galileu.Core;

public class NeuralNetworkModelData
{
    public int InputSize { get; set; }
    public int HiddenSize { get; set; }
    public int OutputSize { get; set; }
    public TensorData WeightsHidden { get; set; }
    public TensorData BiasHidden { get; set; }
    public TensorData WeightsOutput { get; set; }
    public TensorData BiasOutput { get; set; }
}