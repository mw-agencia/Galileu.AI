using Galileu.Node.Interfaces;

namespace Galileu.Node.Brain;



/// <summary>
/// Proxy que carrega tensor do disco apenas quando acessado.
/// Lazy loading para economizar RAM.
/// </summary>
public class DiskTensorProxy : IDisposable
{
    private readonly DiskTensorPool _pool;
    internal readonly TensorMetadata Metadata;
    private readonly IMathEngine _mathEngine;
    private IMathTensor? _cachedTensor; // null = não carregado
    private bool _isDirty = false;
    private bool _disposed = false;

    internal DiskTensorProxy(DiskTensorPool pool, TensorMetadata metadata, IMathEngine mathEngine)
    {
        _pool = pool;
        Metadata = metadata;
        _mathEngine = mathEngine;
    }

    /// <summary>
    /// Obtém o tensor (carrega do disco se necessário).
    /// </summary>
    public IMathTensor GetTensor()
    {
        if (_cachedTensor == null)
        {
            // Carrega do disco
            _cachedTensor = _pool.ReadTensor(Metadata);
        }
        return _cachedTensor;
    }

    /// <summary>
    /// Marca tensor como modificado (será salvo no disco ao retornar).
    /// </summary>
    public void MarkDirty()
    {
        _isDirty = true;
    }

    /// <summary>
    /// Libera tensor da RAM (mas mantém no disco).
    /// </summary>
    public void Unload()
    {
        if (_cachedTensor != null && _isDirty)
        {
            // Salva de volta no disco antes de descarregar
            _pool.WriteTensor(Metadata, _cachedTensor);
            _isDirty = false;
        }
        
        _cachedTensor?.Dispose();
        _cachedTensor = null;
    }

    public void Dispose()
    {
        if (_disposed) return;
        
        // Salva se houver modificações
        if (_isDirty && _cachedTensor != null)
        {
            _pool.WriteTensor(Metadata, _cachedTensor);
        }
        
        // Libera da RAM
        _cachedTensor?.Dispose();
        
        // Devolve ao pool
        _pool.Return(this);
        
        _disposed = true;
    }
}