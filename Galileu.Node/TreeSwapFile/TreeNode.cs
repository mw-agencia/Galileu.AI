using System.Text;

namespace Galileu.Node.TreeSwapFile;

public class TreeNode
{
    public const int MaxDataSize = 1024; // Tamanho máximo de dados para armazenar no próprio nó (bytes)

    // Dados do nó (pode ser o próprio dado ou metadados de disco)
    public byte[] Data { get; set; }
    public int Offset { get; set; } // Offset no arquivo de dados
    public int Length { get; set; } // Comprimento dos dados no arquivo
    public long Id { get; set; } // Identificador único do nó

    public TreeNode Left { get; set; }
    public TreeNode Right { get; set; }

    public TreeNode(long id, byte[] data, int offset = -1, int length = -1)
    {
        Id = id;
        if (offset == -1 && length == -1 && data.Length <= MaxDataSize)
        {
            Data = data; // Armazena dados in-memory
            Offset = -1;
            Length = -1;
        }
        else
        {
            Data = null; // Dados estão no disco
            Offset = offset;
            Length = length;
        }
    }
}