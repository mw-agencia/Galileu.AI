using OpenCL.Net;
using System;

namespace Galileu.Node.Brain.Gpu;

public class GpuTensor : IDisposable
{
    private readonly OpenCLService _service;
    public IMem Buffer { get; }
    public int Rows { get; }
    public int Cols { get; }
    public int TotalSize { get; }

    public GpuTensor(OpenCLService service, int rows, int cols = 1)
    {
        _service = service;
        Rows = rows;
        Cols = cols;
        TotalSize = rows * cols;

        if (!_service.IsGpuAvailable || !_service.Context.HasValue)
        {
            throw new InvalidOperationException("OpenCLService não está disponível para criar um GpuTensor.");
        }

        Buffer = Cl.CreateBuffer(_service.Context.Value, MemFlags.ReadWrite, (IntPtr)(sizeof(double) * TotalSize),
            IntPtr.Zero, out ErrorCode error);
        if (error != ErrorCode.Success) throw new Exception($"Erro ao criar buffer da GPU: {error}");
    }

    public void Write(double[] data)
    {
        if (!_service.CommandQueue.HasValue) return;

        ErrorCode error = Cl.EnqueueWriteBuffer(_service.CommandQueue.Value, Buffer, Bool.True, IntPtr.Zero,
            (IntPtr)(sizeof(double) * data.Length), data, 0, null, out _);
        if (error != ErrorCode.Success) throw new Exception($"Erro ao escrever no buffer da GPU: {error}");
    }

    public double[] Read()
    {
        if (!_service.CommandQueue.HasValue) return Array.Empty<double>();

        var data = new double[TotalSize];
        ErrorCode error = Cl.EnqueueReadBuffer(_service.CommandQueue.Value, Buffer, Bool.True, IntPtr.Zero,
            (IntPtr)(sizeof(double) * TotalSize), data, 0, null, out _);
        if (error != ErrorCode.Success) throw new Exception($"Erro ao ler do buffer da GPU: {error}");
        return data;
    }

    public void Dispose()
    {
        // CORREÇÃO: A liberação de um objeto de memória é feita com Cl.ReleaseMemObject.
        // Verificamos se o buffer não é nulo antes de tentar liberar.
        if (Buffer != null)
        {
            Cl.ReleaseMemObject(Buffer);
        }
    }
}