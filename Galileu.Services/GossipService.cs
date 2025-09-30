using Galileu.Models;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;

namespace Galileu.Services;

public class GossipService : IHostedService, IDisposable
{
    private readonly NodeState _nodeState;
    private readonly NodeClient _nodeClient;
    private readonly ILogger<GossipService> _logger;
    private Timer? _timer;
    private readonly Random _random = new();

    public GossipService(NodeState nodeState, NodeClient nodeClient, ILogger<GossipService> logger)
    {
        _nodeState = nodeState;
        _nodeClient = nodeClient;
        _logger = logger;
    }

    public Task StartAsync(CancellationToken cancellationToken)
    {
        _logger.LogInformation("Gossip Service is starting. Will attempt to gossip every 10 seconds.");
        // O timer começa após 5 segundos e repete a cada 10 segundos.
        _timer = new Timer(DoGossip, null, TimeSpan.FromSeconds(5), TimeSpan.FromSeconds(10));
        return Task.CompletedTask;
    }

    private async void DoGossip(object? state)
    {
        var knownPeers = _nodeState.GetKnownPeers();
        var otherPeers = knownPeers.Where(p => p != _nodeState.Address).ToList();

        if (!otherPeers.Any())
        {
            // Se só conhecemos a nós mesmos, não há com quem fofocar.
            // Isso é um estado normal e esperado para um nó novo ou para o primeiro nó da rede.
            return;
        }
        
        var targetPeerAddress = otherPeers[_random.Next(otherPeers.Count)];

        try
        {
            _logger.LogInformation("Attempting to gossip with peer: {TargetPeer}", targetPeerAddress);

            var request = new GossipSyncRequest(Guid.NewGuid(), knownPeers);
            
            // --- INÍCIO DA CORREÇÃO ---

            // 1. Adicionamos um CancellationToken com timeout.
            //    Isso evita que a tarefa fique pendurada indefinidamente se o outro nó não responder.
            using var cts = new CancellationTokenSource(TimeSpan.FromSeconds(5));
            
            var response = await _nodeClient.SendRequestAsync<GossipSyncResponse>(
                targetPeerAddress, 
                request, 
                cts.Token // Passamos o token com timeout
            );

            // --- FIM DA CORREÇÃO ---

            _nodeState.MergePeers(response.KnownPeers);
            _logger.LogInformation("Gossip with {TargetPeer} successful. Total known peers: {PeerCount}", targetPeerAddress, _nodeState.GetKnownPeers().Count);
            _nodeState.PrintStatus();
        }
        catch (Exception ex)
        {
            // --- INÍCIO DA CORREÇÃO ---

            // 2. Tornamos o log de erro mais informativo.
            _logger.LogWarning(
                "Failed to gossip with peer {TargetPeer}. Reason: {ErrorType} - {ErrorMessage}. Will retry on the next cycle.", 
                targetPeerAddress, 
                ex.GetType().Name, 
                ex.Message);
            
            // --- FIM DA CORREÇÃO ---
        }
    }

    public Task StopAsync(CancellationToken cancellationToken)
    {
        _logger.LogInformation("Gossip Service is stopping.");
        _timer?.Change(Timeout.Infinite, 0);
        return Task.CompletedTask;
    }

    public void Dispose()
    {
        _timer?.Dispose();
    }
}