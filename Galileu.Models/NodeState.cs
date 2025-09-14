using System.Collections.Concurrent;

namespace Galileu.Services;

public class NodeState
{
    public string Id { get; }
    public string Address { get; } // Nosso próprio endereço (ex: http://localhost:5001)

    // Lista thread-safe de endereços dos pares que este nó conhece.
    private readonly ConcurrentDictionary<string, DateTime> _knownPeers = new();
    

    public NodeState(string address)
    {
        Id = Guid.NewGuid().ToString("N");
        Address = address;
        // Um nó sempre se conhece.
        _knownPeers[Address] = DateTime.UtcNow;
    }

    /// <summary>
    /// Retorna uma cópia da lista de pares conhecidos.
    /// </summary>
    public List<string> GetKnownPeers()
    {
        // Retorna apenas os pares que vimos recentemente (ex: nos últimos 5 minutos)
        // para remover automaticamente nós mortos.
        return _knownPeers
            .Where(p => (DateTime.UtcNow - p.Value).TotalMinutes < 5)
            .Select(p => p.Key)
            .ToList();
    }
    
    /// <summary>
    /// Mescla uma lista de pares recebidos com nossa lista local.
    /// </summary>
    public void MergePeers(IEnumerable<string> receivedPeers)
    {
        foreach (var peerAddress in receivedPeers)
        {
            // Adiciona ou atualiza o timestamp do par.
            _knownPeers[peerAddress] = DateTime.UtcNow;
        }
    }

    public void PrintStatus()
    {
        Console.WriteLine("------------------------------------------");
        Console.WriteLine($"| Node ID:    {Id}");
        Console.WriteLine($"| Address:    {Address}");
        Console.WriteLine($"| Known Peers: {GetKnownPeers().Count}");
        // Para depuração: Console.WriteLine(string.Join(", ", GetKnownPeers()));
        Console.WriteLine("------------------------------------------\n");
    }
    public void RemovePeer(string peerAddress)
    {
        if (_knownPeers.TryRemove(peerAddress, out _))
        {
            Console.WriteLine($"[State] Peer {peerAddress} removido da lista devido a falha de health check.");
        }
    }
}