using OpenCL.Net;
using System;
using Galileu.Node.Interfaces;
using Galileu.Node.Core;

namespace Galileu.Node.Brain.Gpu;

public class GpuTensor : IMathTensor
{
    private readonly OpenCLService _service;
    public IMem Buffer { get; }
    public int[] Shape { get; }
    public int TotalSize { get; }

    public GpuTensor(OpenCLService service, int[] shape)
    {
        _service = service;
        Shape = shape;
        TotalSize = 1;
        foreach (int dim in shape) TotalSize *= dim;

        if (!_service.IsGpuAvailable || !_service.Context.HasValue)
            throw new InvalidOperationException("OpenCLService não está disponível.");

        Buffer = Cl.CreateBuffer(_service.Context.Value, MemFlags.ReadWrite, (IntPtr)(sizeof(double) * TotalSize), IntPtr.Zero, out var error);
        if (error != ErrorCode.Success) throw new Exception($"Erro ao criar buffer da GPU: {error}");
    }
    
    public void Write(double[] data)
    {
        if (!_service.CommandQueue.HasValue) return;
        var error = Cl.EnqueueWriteBuffer(_service.CommandQueue.Value, Buffer, Bool.True, IntPtr.Zero, (IntPtr)(sizeof(double) * data.Length), data, 0, null, out _);
        if (error != ErrorCode.Success) throw new Exception($"Erro ao escrever no buffer da GPU: {error}");
    }

    public double[] Read()
    {
        if (!_service.CommandQueue.HasValue) return Array.Empty<double>();
        var data = new double[TotalSize];
        var error = Cl.EnqueueReadBuffer(_service.CommandQueue.Value, Buffer, Bool.True, IntPtr.Zero, (IntPtr)(sizeof(double) * TotalSize), data, 0, null, out _);
        if (error != ErrorCode.Success) throw new Exception($"Erro ao ler do buffer da GPU: {error}");
        return data;
    }

    public Tensor ToCpuTensor()
    {
        return new Tensor(Read(), Shape);
    }

    public void Dispose()
    {
        if (Buffer != null)
        {
            Cl.ReleaseMemObject(Buffer);
        }
        GC.SuppressFinalize(this);
    }
}