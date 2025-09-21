using System.Text;

namespace Galileu.Node.TreeSwapFile;

public class BinaryTreeFileStorage : IDisposable
{
    private readonly string _filePath;
    private readonly ReaderWriterLockSlim _lock = new ReaderWriterLockSlim();
    private long _nextId = 0;
    private long _currentFilePosition = 0;
    private TreeNode _root;

    public BinaryTreeFileStorage(string filePath)
    {
        _filePath = filePath;

        var directory = Path.GetDirectoryName(_filePath);
        if (directory != null && !Directory.Exists(directory))
        {
            Directory.CreateDirectory(directory);
        }
    }

    // Escreve um único dado usando um FileStream já aberto.
    public long StoreData(string data, FileStream fileStream)
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
                fileStream.Seek(_currentFilePosition, SeekOrigin.Begin);
                fileStream.Write(bytes, 0, bytes.Length);
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

    // Lê um único dado, abrindo e fechando o arquivo sob demanda.
    public string GetData(long id)
    {
        _lock.EnterReadLock();
        try
        {
            TreeNode node = FindNode(_root, id);
            if (node == null) throw new KeyNotFoundException($"Dados com ID {id} não encontrados.");

            if (node.Data != null)
            {
                return Encoding.UTF8.GetString(node.Data);
            }

            using (var fileStream = new FileStream(_filePath, FileMode.Open, FileAccess.Read, FileShare.ReadWrite))
            {
                fileStream.Seek(node.Offset, SeekOrigin.Begin);
                byte[] bytes = new byte[node.Length];
                fileStream.Read(bytes, 0, node.Length);
                return Encoding.UTF8.GetString(bytes);
            }
        }
        finally
        {
            _lock.ExitReadLock();
        }
    }

    public void Dispose()
    {
        _lock?.Dispose();
    }

    private TreeNode InsertNode(TreeNode current, TreeNode newNode)
    {
        if (current == null) return newNode;
        if (newNode.Id < current.Id) current.Left = InsertNode(current.Left, newNode);
        else current.Right = InsertNode(current.Right, newNode);
        return current;
    }

    private TreeNode FindNode(TreeNode current, long id)
    {
        if (current == null || current.Id == id) return current;
        if (id < current.Id) return FindNode(current.Left, id);
        else return FindNode(current.Right, id);
    }
}