// --- START OF FILE TreeSwapFile/BinaryTreeFileStorage.cs (FINAL CORRECTED VERSION) ---

using System;
using System.IO;
using System.Text;
using System.Threading;

namespace Galileu.Node.TreeSwapFile;

public class BinaryTreeFileStorage : IDisposable
{
    private readonly string _filePath;
    private readonly FileStream _fileStream;
    private readonly ReaderWriterLockSlim _lock = new ReaderWriterLockSlim();
    private bool _disposed = false;
    private static readonly Encoding Utf8NoBom = new UTF8Encoding(false);

    public BinaryTreeFileStorage(string filePath)
    {
        _filePath = filePath;
        var directory = Path.GetDirectoryName(_filePath);
        if (directory != null && !Directory.Exists(directory))
        {
            Directory.CreateDirectory(directory);
        }
        // Abre o arquivo uma vez e o mantém aberto
        _fileStream = new FileStream(_filePath, FileMode.OpenOrCreate, FileAccess.ReadWrite, FileShare.Read);
    }

    // O StoreData agora é para armazenamento sequencial
    public long StoreData(string data)
    {
        _lock.EnterWriteLock();
        try
        {
            if (_disposed) throw new ObjectDisposedException(nameof(BinaryTreeFileStorage));

            byte[] dataBytes = Utf8NoBom.GetBytes(data);
            if (dataBytes.Length > TreeNode.MaxDataSize)
            {
                // Ignora dados que são grandes demais para o bloco
                Console.WriteLine($"Aviso: Amostra com {dataBytes.Length} bytes excedeu o máximo de {TreeNode.MaxDataSize} e foi ignorada.");
                return -1;
            }

            var node = new TreeNode();
            Array.Copy(dataBytes, node.Data, dataBytes.Length);

            long offset = _fileStream.Length; // O offset é a posição atual do final do arquivo
            _fileStream.Seek(offset, SeekOrigin.Begin);
            
            byte[] nodeBytes = node.Serialize();
            _fileStream.Write(nodeBytes, 0, nodeBytes.Length);
            
            return offset;
        }
        finally
        {
            _lock.ExitWriteLock();
        }
    }

    // GetData agora lê um bloco de tamanho fixo em um offset específico
    public string GetData(long offset)
    {
        _lock.EnterReadLock();
        try
        {
            if (_disposed) throw new ObjectDisposedException(nameof(BinaryTreeFileStorage));
            
            byte[] nodeBuffer = new byte[TreeNode.NodeSize];
            _fileStream.Seek(offset, SeekOrigin.Begin);
            int bytesRead = _fileStream.Read(nodeBuffer, 0, TreeNode.NodeSize);

            if (bytesRead != TreeNode.NodeSize)
                throw new IOException($"Leitura incompleta no offset {offset}.");
            
            var node = TreeNode.Deserialize(nodeBuffer);
            
            // Converte os dados para string, removendo os bytes nulos do final
            return Utf8NoBom.GetString(node.Data).TrimEnd('\0');
        }
        finally
        {
            _lock.ExitReadLock();
        }
    }

    public void Dispose()
    {
        Dispose(true);
        GC.SuppressFinalize(this);
    }

    protected virtual void Dispose(bool disposing)
    {
        if (_disposed) return;
        if (disposing)
        {
            _lock.EnterWriteLock();
            try
            {
                _fileStream?.Close();
                _fileStream?.Dispose();
            }
            finally
            {
                _lock.ExitWriteLock();
            }
            _lock?.Dispose();

            try { if (File.Exists(_filePath)) File.Delete(_filePath); }
            catch (IOException) { /* Ignora */ }
        }
        _disposed = true;
    }

    public void Clear()
    {
        _lock.EnterWriteLock();
        try
        {
            if (_disposed) return;
            _fileStream.SetLength(0); // Trunca o arquivo para tamanho zero
            _fileStream.Flush();
        }
        finally
        {
            _lock.ExitWriteLock();
        }
    }
}