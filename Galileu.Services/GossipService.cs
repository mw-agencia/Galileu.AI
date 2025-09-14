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
        _logger.LogInformation("Gossip Service is starting.");
        // Inicia o timer para fofocar a cada 10 segundos com um atraso inicial.
        _timer = new Timer(DoGossip, null, TimeSpan.FromSeconds(5), TimeSpan.FromSeconds(10));
        return Task.CompletedTask;
    }

    private async void DoGossip(object? state)
    {
        try
        {
            var knownPeers = _nodeState.GetKnownPeers();
            // Não fofocamos com nós mesmos.
            var otherPeers = knownPeers.Where(p => p != _nodeState.Address).ToList();

            if (!otherPeers.Any())
            {
                //_logger.LogInformation("No other peers to gossip with yet.");
                return;
            }

            // Seleciona um par aleatório para fofocar.
            var targetPeerAddress = otherPeers[_random.Next(otherPeers.Count)];

            _logger.LogInformation("Gossiping with peer: {TargetPeer}", targetPeerAddress);

            var request = new GossipSyncRequest(Guid.NewGuid(), knownPeers);
            
            // Envia nossa lista e recebe a lista do par em troca.
            var response = await _nodeClient.SendRequestAsync<GossipSyncResponse>(
                targetPeerAddress, 
                request, 
                CancellationToken.None // Use CancellationToken com timeout em produção
            );

            // Mescla a lista de pares recebida com a nossa.
            _nodeState.MergePeers(response.KnownPeers);

            _logger.LogInformation("Gossip successful. Total known peers: {PeerCount}", _nodeState.GetKnownPeers().Count);
            _nodeState.PrintStatus();
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "Failed to gossip with a peer.");
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