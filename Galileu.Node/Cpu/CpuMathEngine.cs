using System;
using Galileu.Node.Core;
using Galileu.Node.Interfaces;

namespace Galileu.Node.Cpu;

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
        // Retorna um clone para garantir que o tensor tenha sua própria cópia dos dados.
        return new CpuTensor((double[])hostData.Clone(), shape);
    }

    #region Operações do Forward Pass

    public void Add(IMathTensor a, IMathTensor b, IMathTensor result)
    {
        var A = ((CpuTensor)a).GetData();
        var B = ((CpuTensor)b).GetData();
        var C = ((CpuTensor)result).GetData();
        for (int i = 0; i < A.Length; i++) C[i] = A[i] + B[i];
    }

    public void AddBroadcast(IMathTensor matrix, IMathTensor vector, IMathTensor result)
    {
        var A = ((CpuTensor)matrix).GetData();
        var B = ((CpuTensor)vector).GetData();
        var C = ((CpuTensor)result).GetData();
        int M = matrix.Shape[0];
        int N = matrix.Shape[1];
        for (int i = 0; i < M; i++)
        {
            for (int j = 0; j < N; j++) C[i * N + j] = A[i * N + j] + B[j];
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
        Array.Clear(C, 0, C.Length);
        for (int i = 0; i < M; i++)
        {
            for (int j = 0; j < N; j++)
            {
                double sum = 0;
                for (int k = 0; k < K; k++) sum += A[i * K + k] * B[k * N + j];
                C[i * N + j] = sum;
            }
        }
    }

    public void Multiply(IMathTensor a, IMathTensor b, IMathTensor result)
    {
        var A = ((CpuTensor)a).GetData();
        var B = ((CpuTensor)b).GetData();
        var C = ((CpuTensor)result).GetData();
        for (int i = 0; i < A.Length; i++) C[i] = A[i] * B[i];
    }

    public void Sigmoid(IMathTensor input, IMathTensor result)
    {
        var I = ((CpuTensor)input).GetData();
        var O = ((CpuTensor)result).GetData();
        for (int i = 0; i < I.Length; i++) O[i] = 1.0 / (1.0 + Math.Exp(-I[i]));
    }

    public void Tanh(IMathTensor input, IMathTensor result)
    {
        var I = ((CpuTensor)input).GetData();
        var O = ((CpuTensor)result).GetData();
        for (int i = 0; i < I.Length; i++) O[i] = Math.Tanh(I[i]);
    }

    #endregion

    #region Operações do Backward Pass (BPTT) e Utilitários

    public IMathTensor Clone(IMathTensor tensor)
    {
        var cpuTensor = (CpuTensor)tensor;
        return new CpuTensor((double[])cpuTensor.GetData().Clone(), (int[])cpuTensor.Shape.Clone());
    }

    public void Transpose(IMathTensor input, IMathTensor result)
    {
        var I = ((CpuTensor)input).GetData();
        var O = ((CpuTensor)result).GetData();
        int rows = input.Shape[0];
        int cols = input.Shape[1];
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++) O[j * rows + i] = I[i * cols + j];
        }
    }

    public void Subtract(IMathTensor a, IMathTensor b, IMathTensor result)
    {
        var A = ((CpuTensor)a).GetData();
        var B = ((CpuTensor)b).GetData();
        var C = ((CpuTensor)result).GetData();
        for (int i = 0; i < A.Length; i++) C[i] = A[i] - B[i];
    }

    public void SigmoidDerivative(IMathTensor output, IMathTensor result)
    {
        var O = ((CpuTensor)output).GetData();
        var R = ((CpuTensor)result).GetData();
        for (int i = 0; i < O.Length; i++) R[i] = O[i] * (1.0 - O[i]);
    }

    public void TanhDerivative(IMathTensor output, IMathTensor result)
    {
        var O = ((CpuTensor)output).GetData();
        var R = ((CpuTensor)result).GetData();
        for (int i = 0; i < O.Length; i++) R[i] = 1.0 - O[i] * O[i];
    }

    public void MatrixMultiplyTransposeA(IMathTensor a, IMathTensor b, IMathTensor result)
    {
        // Calcula A' * B, onde A' é a transposta de A.
        // A: [K, M], A': [M, K]
        // B: [K, N]
        // Result C: [M, N]
        var A = ((CpuTensor)a).GetData();
        var B = ((CpuTensor)b).GetData();
        var C = ((CpuTensor)result).GetData();
        int K = a.Shape[0];
        int M = a.Shape[1];
        int N = b.Shape[1];
        Array.Clear(C, 0, C.Length);
        for (int i = 0; i < M; i++) // Linhas de C (e linhas de A')
        {
            for (int j = 0; j < N; j++) // Colunas de C (e colunas de B)
            {
                double sum = 0;
                for (int k = 0; k < K; k++) // Colunas de A' (e linhas de B)
                {
                    sum += A[k * M + i] * B[k * N + j];
                }
                C[i * N + j] = sum; // Acumula gradientes
            }
        }
    }

    public void MatrixMultiplyTransposeB(IMathTensor a, IMathTensor b, IMathTensor result)
    {
        // Calcula A * B', onde B' é a transposta de B.
        // A: [M, K]
        // B: [N, K], B': [K, N]
        // Result C: [M, N]
        var A = ((CpuTensor)a).GetData();
        var B = ((CpuTensor)b).GetData();
        var C = ((CpuTensor)result).GetData();
        int M = a.Shape[0];
        int K = a.Shape[1];
        int N = b.Shape[0];
        Array.Clear(C, 0, C.Length);
        for (int i = 0; i < M; i++) // Linhas de C (e linhas de A)
        {
            for (int j = 0; j < N; j++) // Colunas de C (e linhas de B)
            {
                double sum = 0;
                for (int k = 0; k < K; k++) // Colunas de A (e colunas de B)
                {
                    sum += A[i * K + k] * B[j * K + k];
                }
                C[i * N + j] = sum;
            }
        }
    }

    public void AddScaled(IMathTensor target, IMathTensor source, double scalar)
    {
        var T = ((CpuTensor)target).GetData();
        var S = ((CpuTensor)source).GetData();
        for (int i = 0; i < T.Length; i++) T[i] += S[i] * scalar;
    }

    public void SubtractScaled(IMathTensor target, IMathTensor source, double scalar)
    {
        var T = ((CpuTensor)target).GetData();
        var S = ((CpuTensor)source).GetData();
        for (int i = 0; i < T.Length; i++) T[i] -= S[i] * scalar;
    }
    
    public void Slice(IMathTensor source, int rowIndex, IMathTensor destination)
    {
        var srcData = ((CpuTensor)source).GetData();
        var destData = ((CpuTensor)destination).GetData();
        int featureSize = destination.Shape[1];
        int offset = rowIndex * featureSize;
        Array.Copy(srcData, offset, destData, 0, featureSize);
    }

    public void Set(IMathTensor destination, int rowIndex, IMathTensor source)
    {
        var srcData = ((CpuTensor)source).GetData();
        var destData = ((CpuTensor)destination).GetData();
        int featureSize = source.Shape[1];
        int offset = rowIndex * featureSize;
        Array.Copy(srcData, 0, destData, offset, featureSize);
    }

    public void Clip(IMathTensor tensor, double minValue, double maxValue)
    {
        var data = ((CpuTensor)tensor).GetData();
        for (int i = 0; i < data.Length; i++)
        {
            if (data[i] < minValue) data[i] = minValue;
            else if (data[i] > maxValue) data[i] = maxValue;
        }
    }
    
    public void Scale(IMathTensor tensor, double scalar)
    {
        var data = ((CpuTensor)tensor).GetData();
        for (int i = 0; i < data.Length; i++)
        {
            data[i] *= scalar;
        }
    }
    
    public void Softmax(IMathTensor input, IMathTensor result)
    {
        var inputData = ((CpuTensor)input).GetData();
        var outputData = ((CpuTensor)result).GetData();
        int rows = input.Shape[0];
        int cols = input.Shape[1];
    
        for (int row = 0; row < rows; row++)
        {
            int offset = row * cols;
        
            // Encontra máximo para estabilidade numérica
            double maxVal = inputData[offset];
            for (int i = 1; i < cols; i++)
            {
                if (inputData[offset + i] > maxVal)
                    maxVal = inputData[offset + i];
            }
        
            // Calcula exp e soma
            double sumExp = 0;
            for (int i = 0; i < cols; i++)
            {
                outputData[offset + i] = Math.Exp(inputData[offset + i] - maxVal);
                sumExp += outputData[offset + i];
            }
        
            // Normaliza
            for (int i = 0; i < cols; i++)
            {
                outputData[offset + i] /= sumExp;
            }
        }
    }
    
    public void Lookup(IMathTensor embeddingMatrix, int index, IMathTensor result)
    {
        var matrixData = ((CpuTensor)embeddingMatrix).GetData();
        var resultData = ((CpuTensor)result).GetData();
        int embeddingSize = embeddingMatrix.Shape[1];
        int offset = index * embeddingSize;
        Array.Copy(matrixData, offset, resultData, 0, embeddingSize);
    }

    public void AccumulateGradient(IMathTensor embeddingGradients, IMathTensor gradient, int index)
    {
        var embeddingGradData = ((CpuTensor)embeddingGradients).GetData();
        var gradData = ((CpuTensor)gradient).GetData();
        int embeddingSize = embeddingGradients.Shape[1];
        int offset = index * embeddingSize;
        for (int i = 0; i < embeddingSize; i++)
        {
            embeddingGradData[offset + i] += gradData[i];
        }
    }
    
    public void SoftmaxCrossEntropyGradient(IMathTensor predictions, int[] targetIndices, IMathTensor result)
    {
        var predData = ((CpuTensor)predictions).GetData();
        var resultData = ((CpuTensor)result).GetData();
        int sequenceLength = predictions.Shape[0];
        int vocabSize = predictions.Shape[1];

        // Passo 1: Copia as predições para o tensor de resultado (dy = p)
        Array.Copy(predData, resultData, predData.Length);

        // Passo 2: Subtrai 1 na posição do índice correto para cada item da sequência (dy = p - y)
        for (int t = 0; t < sequenceLength; t++)
        {
            int targetIndex = targetIndices[t];
            int flatIndex = t * vocabSize + targetIndex;
            resultData[flatIndex] -= 1.0;
        }
    }
    
    // --- IMPLEMENTAÇÃO DO NOVO MÉTODO ---
    public void Copy(IMathTensor source, IMathTensor destination)
    {
        if (source.Length != destination.Length)
            throw new ArgumentException("Os tensores de origem e destino devem ter o mesmo tamanho para a cópia.");

        var srcData = ((CpuTensor)source).GetData();
        var destData = ((CpuTensor)destination).GetData();
        Array.Copy(srcData, destData, srcData.Length);
    }

    #endregion

    public void Dispose()
    {
        // A engine de CPU não gerencia recursos não gerenciados (como handles de GPU),
        // então não há nada a ser feito aqui. O Garbage Collector do .NET cuida da memória.
    }
}

  