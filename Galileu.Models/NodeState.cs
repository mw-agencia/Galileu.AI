namespace Galileu.Models;


public class NodeState
{
    public string Id { get; }
    public string Address { get; }

    private readonly object _lock = new();

    private string? _parentAddress;
    private string? _leftChildAddress;
    private string? _rightChildAddress;

    public NodeState(string address)
    {
        Id = Guid.NewGuid().ToString("N");
        Address = address;
    }

    public string? ParentAddress
    {
        get { lock (_lock) return _parentAddress; }
        set { lock (_lock) _parentAddress = value; }
    }

    public string? LeftChildAddress
    {
        get { lock (_lock) return _leftChildAddress; }
        set { lock (_lock) _leftChildAddress = value; }
    }

    public string? RightChildAddress
    {
        get { lock (_lock) return _rightChildAddress; }
        set { lock (_lock) _rightChildAddress = value; }
    }

    public void PrintStatus()
    {
        lock (_lock)
        {
            Console.WriteLine("------------------------------------------");
            Console.WriteLine($"| Node ID:    {Id}");
            Console.WriteLine($"| Address:    {Address}");
            Console.WriteLine($"| Parent:     {ParentAddress ?? "None (I am the ROOT)"}");
            Console.WriteLine($"| Left Child: {LeftChildAddress ?? "None"}");
            Console.WriteLine($"| Right Child:{RightChildAddress ?? "None"}");
            Console.WriteLine("------------------------------------------\n");
        }
    }
}