namespace Galileu.Node.Brain;

public class AdamOptimizer
{
    private readonly double _learningRate;
    private readonly double _beta1;
    private readonly double _beta2;
    private readonly double _epsilon;

    // Dicionários para guardar momentos e timesteps PARA CADA camada/parâmetro
    private readonly Dictionary<int, double[]> _m;
    private readonly Dictionary<int, double[]> _v;
    private readonly Dictionary<int, int> _t; 

    public AdamOptimizer(double learningRate = 0.001, double beta1 = 0.9, 
        double beta2 = 0.999, double epsilon = 1e-8)
    {
        _learningRate = learningRate;
        _beta1 = beta1;
        _beta2 = beta2;
        _epsilon = epsilon;
        _m = new Dictionary<int, double[]>();
        _v = new Dictionary<int, double[]>();
        _t = new Dictionary<int, int>(); 
    }

    public void UpdateParameters(int layerId, double[] parameters, double[] gradients)
    {
        // Inicializa se for a primeira vez que vemos esta camada
        if (!_m.ContainsKey(layerId))
        {
            _m[layerId] = new double[parameters.Length];
            _v[layerId] = new double[parameters.Length];
            _t[layerId] = 0; 
        }

        // Incrementa o timestep para ESTA camada específica
        _t[layerId]++; 
        int t = _t[layerId]; 

        var m = _m[layerId];
        var v = _v[layerId];

        for (int i = 0; i < parameters.Length; i++)
        {
            double g = gradients[i];

            // Atualiza a estimativa do primeiro momento
            m[i] = _beta1 * m[i] + (1 - _beta1) * g;
            // Atualiza a estimativa do segundo momento
            v[i] = _beta2 * v[i] + (1 - _beta2) * (g * g);

            // Calcula a correção de viés para o primeiro momento
            double mHat = m[i] / (1 - Math.Pow(_beta1, t)); 
            // Calcula a correção de viés para o segundo momento
            double vHat = v[i] / (1 - Math.Pow(_beta2, t)); 

            // Atualiza os parâmetros
            parameters[i] -= _learningRate * mHat / (Math.Sqrt(vHat) + _epsilon);
        }
    }
}