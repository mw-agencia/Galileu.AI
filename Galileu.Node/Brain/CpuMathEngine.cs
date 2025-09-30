using System;
using Galileu.Node.Core;
using Galileu.Node.Interfaces;

namespace Galileu.Node.Brain;

public class CpuMathEngine : IMathEngine
{
    public bool IsGpu => false;

    public IMathTensor CreateTensor(int[] shape)
    {
        int size = 1;
        foreach (int dim in shape) size *= dim;
        return new CpuTensor(new double[size], shape);
    }

    public IMathTensor CreateTensor(double[] hostData, int[] shape)
    {
        // Clona os dados para garantir que o tensor tenha sua própria cópia.
        return new CpuTensor((double[])hostData.Clone(), shape);
    }

    public void Add(IMathTensor a, IMathTensor b, IMathTensor result)
    {
        var A = ((CpuTensor)a).GetData();
        var B = ((CpuTensor)b).GetData();
        var C = ((CpuTensor)result).GetData();

        for (int i = 0; i < A.Length; i++)
        {
            C[i] = A[i] + B[i];
        }
    }

    public void AddBroadcast(IMathTensor matrix, IMathTensor vector, IMathTensor result)
    {
        var A = ((CpuTensor)matrix).GetData();
        var B = ((CpuTensor)vector).GetData();
        var C = ((CpuTensor)result).GetData();

        int M = matrix.Shape[0]; // Linhas da matriz
        int N = matrix.Shape[1]; // Colunas da matriz e tamanho do vetor

        for (int i = 0; i < M; i++)
        {
            for (int j = 0; j < N; j++)
            {
                C[i * N + j] = A[i * N + j] + B[j];
            }
        }
    }

    public void MatrixMultiply(IMathTensor a, IMathTensor b, IMathTensor result)
    {
        var A = ((CpuTensor)a).GetData();
        var B = ((CpuTensor)b).GetData();
        var C = ((CpuTensor)result).GetData();

        int M = a.Shape[0];
        int K = a.Shape[1];
        int N = b.Shape[1];

        // É crucial zerar o tensor de resultado antes da multiplicação.
        Array.Clear(C, 0, C.Length);

        for (int i = 0; i < M; i++)
        {
            for (int j = 0; j < N; j++)
            {
                double sum = 0;
                for (int k = 0; k < K; k++)
                {
                    sum += A[i * K + k] * B[k * N + j];
                }

                C[i * N + j] = sum;
            }
        }
    }

    public void Multiply(IMathTensor a, IMathTensor b, IMathTensor result)
    {
        var A = ((CpuTensor)a).GetData();
        var B = ((CpuTensor)b).GetData();
        var C = ((CpuTensor)result).GetData();

        for (int i = 0; i < A.Length; i++)
        {
            C[i] = A[i] * B[i];
        }
    }

    public void Sigmoid(IMathTensor input, IMathTensor result)
    {
        var I = ((CpuTensor)input).GetData();
        var O = ((CpuTensor)result).GetData();
        for (int i = 0; i < I.Length; i++)
        {
            O[i] = 1.0 / (1.0 + Math.Exp(-I[i]));
        }
    }

    public void Tanh(IMathTensor input, IMathTensor result)
    {
        var I = ((CpuTensor)input).GetData();
        var O = ((CpuTensor)result).GetData();
        for (int i = 0; i < I.Length; i++)
        {
            O[i] = Math.Tanh(I[i]);
        }
    }

    public void Dispose()
    {
        
    }
}

  