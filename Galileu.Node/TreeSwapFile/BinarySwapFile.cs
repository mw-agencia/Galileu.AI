using System.Text;

namespace Galileu.Node.TreeSwapFile;

public class BinarySwapFile : IDisposable
{
    private readonly string _filePath;
    private readonly FileStream _fileStream;
    private readonly ReaderWriterLockSlim _lock = new ReaderWriterLockSlim();
    private long _nextId = 0; // Para gerar IDs únicos para os nós
    private long _currentFilePosition = 0; // Posição atual de escrita no arquivo

    private TreeNode _root; // Raiz da árvore de nós

    public BinarySwapFile(string filePath)
    {
        _filePath = filePath;
        _fileStream = new FileStream(filePath, FileMode.OpenOrCreate, FileAccess.ReadWrite, FileShare.None);
    }

    public long StoreData(string data)
    {
        _lock.EnterWriteLock();
        try
        {
            long id = Interlocked.Increment(ref _nextId);
            byte[] bytes = Encoding.UTF8.GetBytes(data);

            TreeNode newNode;
            if (bytes.Length <= TreeNode.MaxDataSize)
            {
                newNode = new TreeNode(id, bytes); // In-memory
            }
            else
            {
                _fileStream.Seek(_currentFilePosition, SeekOrigin.Begin);
                _fileStream.Write(bytes, 0, bytes.Length);
                newNode = new TreeNode(id, null, (int)_currentFilePosition, bytes.Length);
                _currentFilePosition += bytes.Length;
            }

            _root = InsertNode(_root, newNode);
            return id;
        }
        finally
        {
            _lock.ExitWriteLock();
        }
    }

    // Armazena um fluxo de amostras (strings JSON) e retorna seus offsets/IDs
    public List<long> StreamAndStoreSamples(IEnumerable<string> samplesStream, Action<int> progressCallback = null)
    {
        var offsets = new List<long>();
        int count = 0;
        foreach (var sampleJson in samplesStream)
        {
            offsets.Add(StoreData(sampleJson));
            count++;
            progressCallback?.Invoke(count);
        }

        return offsets;
    }

    public string GetData(long id)
    {
        _lock.EnterReadLock();
        try
        {
            TreeNode node = FindNode(_root, id);
            if (node == null)
            {
                throw new KeyNotFoundException($"Dados com ID {id} não encontrados.");
            }

            byte[] bytes;
            if (node.Data != null)
            {
                bytes = node.Data; // Dados in-memory
            }
            else
            {
                _fileStream.Seek(node.Offset, SeekOrigin.Begin);
                bytes = new byte[node.Length];
                _fileStream.Read(bytes, 0, node.Length);
            }

            return Encoding.UTF8.GetString(bytes);
        }
        finally
        {
            _lock.ExitReadLock();
        }
    }

    private TreeNode InsertNode(TreeNode current, TreeNode newNode)
    {
        if (current == null)
        {
            return newNode;
        }

        if (newNode.Id < current.Id)
        {
            current.Left = InsertNode(current.Left, newNode);
        }
        else
        {
            current.Right = InsertNode(current.Right, newNode);
        }

        return current;
    }

    private TreeNode FindNode(TreeNode current, long id)
    {
        if (current == null || current.Id == id)
        {
            return current;
        }

        if (id < current.Id)
        {
            return FindNode(current.Left, id);
        }
        else
        {
            return FindNode(current.Right, id);
        }
    }

    public void Dispose()
    {
        _fileStream?.Dispose();
        _lock?.Dispose();
        // Opcional: deletar o arquivo físico se for temporário
        // File.Delete(_filePath);
    }
}