using System;

namespace Galileu.Node.Brain;

/// <summary>
/// Implementação "stateless" (sem estado) do otimizador Adam.
/// Esta classe contém apenas a lógica matemática para a atualização dos parâmetros.
/// Todo o estado (momentos m, v e timestep t) é gerenciado externamente,
/// permitindo que seja armazenado em disco para economizar RAM.
/// </summary>
public class AdamOptimizer
{
    /// <summary>
    /// Atualiza um conjunto de parâmetros de acordo com a regra do otimizador Adam.
    /// </summary>
    /// <param name="parameters">O array de pesos/biases a ser atualizado (modificado in-place).</param>
    /// <param name="gradients">Os gradientes calculados para estes parâmetros.</param>
    /// <param name="m">O array da estimativa do primeiro momento (modificado in-place).</param>
    /// <param name="v">O array da estimativa do segundo momento (modificado in-place).</param>
    /// <param name="t">O timestep atual (modificado por referência).</param>
    /// <param name="learningRate">A taxa de aprendizado.</param>
    /// <param name="beta1">O fator de decaimento para o primeiro momento.</param>
    /// <param name="beta2">O fator de decaimento para o segundo momento.</param>
    /// <param name="epsilon">Um pequeno valor para evitar divisão por zero.</param>
    public static void UpdateParameters(
        double[] parameters, 
        double[] gradients, 
        double[] m, 
        double[] v, 
        ref int t,
        double learningRate = 0.001, 
        double beta1 = 0.9, 
        double beta2 = 0.999, 
        double epsilon = 1e-8)
    {
        // Incrementa o timestep
        t++; 

        // Calcula as correções de viés (bias correction)
        double beta1_t = 1.0 - Math.Pow(beta1, t);
        double beta2_t = 1.0 - Math.Pow(beta2, t);

        for (int i = 0; i < parameters.Length; i++)
        {
            double g = gradients[i];

            // Atualiza a estimativa do primeiro momento (moving average of gradients)
            m[i] = beta1 * m[i] + (1.0 - beta1) * g;
            
            // Atualiza a estimativa do segundo momento (moving average of squared gradients)
            v[i] = beta2 * v[i] + (1.0 - beta2) * (g * g);

            // Calcula as estimativas corrigidas
            double mHat = m[i] / beta1_t; 
            double vHat = v[i] / beta2_t; 

            // Atualiza os parâmetros
            parameters[i] -= learningRate * mHat / (Math.Sqrt(vHat) + epsilon);
        }
    }
}