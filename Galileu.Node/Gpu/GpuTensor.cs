using System;
using Galileu.Node.Interfaces;
using Galileu.Node.Core;
using OpenCL.NetCore;
using static OpenCL.NetCore.Cl;

namespace Galileu.Node.Gpu;

public class GpuTensor : IMathTensor
{
    public int[] Shape { get; }
    public long Length { get; }
    public bool IsGpu => true;

    // USA O TIPO CONCRETO 'Mem' (struct)
    internal Mem Buffer { get; private set; }

    // USA OS TIPOS CONCRETOS 'Context' E 'CommandQueue'
    private readonly Context _context;
    private readonly CommandQueue _commandQueue;
    private bool _disposed = false;

    public GpuTensor(int[] shape, Context context, CommandQueue commandQueue)
    {
        Shape = shape;
        Length = shape.Aggregate(1L, (a, b) => a * b);
        _context = context;
        _commandQueue = commandQueue;

        ErrorCode error;
        // CORREÇÃO PRINCIPAL: Cast explícito de IMem para Mem.
        Buffer = (Mem)CreateBuffer(_context, MemFlags.ReadWrite, (IntPtr)(Length * sizeof(float)), IntPtr.Zero,
            out error);
        if (error != ErrorCode.Success) throw new OpenClException("Falha ao alocar buffer do tensor.", error);
    }

    public GpuTensor(double[] data, int[] shape, Context context, CommandQueue commandQueue)
    {
        Shape = shape;
        Length = data.Length;
        _context = context;
        _commandQueue = commandQueue;
        var floatData = Array.ConvertAll(data, d => (float)d);

        ErrorCode error;
        // CORREÇÃO PRINCIPAL: Cast explícito de IMem para Mem.
        Buffer = (Mem)CreateBuffer(_context, MemFlags.ReadWrite | MemFlags.CopyHostPtr,
            (IntPtr)(Length * sizeof(float)), floatData, out error);
        if (error != ErrorCode.Success) throw new OpenClException("Falha ao alocar e copiar buffer do tensor.", error);
    }

    public Tensor ToCpuTensor()
    {
        var floatData = new float[Length];
        // Os métodos Enqueue aceitam 'Mem' porque 'Mem' implementa 'IMem'. Isso funciona.
        ErrorCode error = EnqueueReadBuffer(_commandQueue, Buffer, Bool.True, IntPtr.Zero,
            (IntPtr)(Length * sizeof(float)), floatData, 0, null, out _);
        if (error != ErrorCode.Success) throw new OpenClException("Falha ao ler dados da GPU para a CPU.", error);

        var doubleData = Array.ConvertAll(floatData, f => (double)f);
        return new Tensor(doubleData, Shape);
    }

    public void UpdateFromCpu(double[] data)
    {
        var floatData = Array.ConvertAll(data, d => (float)d);
        ErrorCode error = EnqueueWriteBuffer(_commandQueue, Buffer, Bool.True, IntPtr.Zero,
            (IntPtr)(Length * sizeof(float)), floatData, 0, null, out _);
        if (error != ErrorCode.Success) throw new OpenClException("Falha ao escrever dados da CPU para a GPU.", error);
    }

    public void Dispose()
    {
        if (_disposed) return;
        ReleaseMemObject(Buffer);

        _disposed = true;
        GC.SuppressFinalize(this);
    }
}