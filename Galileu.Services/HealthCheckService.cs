// Galileu.Services/HealthCheckService.cs
using Galileu.Models;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;

namespace Galileu.Services;

/// <summary>
/// Serviço de background que ativamente verifica a saúde de uma amostra
/// de pares conhecidos para limpar a lista de nós mortos mais rapidamente.
/// </summary>
public class HealthCheckService : BackgroundService
{
    private readonly NodeState _nodeState;
    private readonly NodeClient _nodeClient;
    private readonly ILogger<HealthCheckService> _logger;
    private readonly TimeSpan _checkInterval = TimeSpan.FromSeconds(30); // A verificação pode ser menos frequente que a fofoca
    private readonly Random _random = new();

    public HealthCheckService(NodeState nodeState, NodeClient nodeClient, ILogger<HealthCheckService> logger)
    {
        _nodeState = nodeState;
        _nodeClient = nodeClient;
        _logger = logger;
    }

    protected override async Task ExecuteAsync(CancellationToken stoppingToken)
    {
        // Aguarda um pouco antes de iniciar o primeiro ciclo para a rede se estabilizar
        await Task.Delay(TimeSpan.FromSeconds(20), stoppingToken);

        while (!stoppingToken.IsCancellationRequested)
        {
            await PerformHealthChecks(stoppingToken);
            await Task.Delay(_checkInterval, stoppingToken);
        }
    }

    private async Task PerformHealthChecks(CancellationToken token)
    {
        var allPeers = _nodeState.GetKnownPeers().Where(p => p != _nodeState.Address).ToList();

        if (!allPeers.Any())
        {
            return; // Nada a fazer se não conhecemos outros nós
        }

        // --- Lógica de Amostragem ---
        // Para evitar sobrecarregar a rede, não pingamos todos os nós sempre.
        // Pingamos uma pequena amostra (ex: até 3 nós aleatórios).
        const int sampleSize = 3;
        var peersToPing = allPeers.OrderBy(x => _random.Next()).Take(sampleSize).ToList();

        _logger.LogInformation("Performing health check on {Count} random peers...", peersToPing.Count);

        var tasks = peersToPing.Select(peerAddress => PingPeerAsync(peerAddress, token));
        await Task.WhenAll(tasks);
    }

    private async Task PingPeerAsync(string peerAddress, CancellationToken token)
    {
        try
        {
            var ping = new PingRequest(Guid.NewGuid(), _nodeState.Id);
            
            // Enviamos o ping com um timeout. Se demorar mais de 5s, consideramos falho.
            using var timeoutCts = CancellationTokenSource.CreateLinkedTokenSource(token);
            timeoutCts.CancelAfter(TimeSpan.FromSeconds(5));

            var response = await _nodeClient.SendRequestAsync<PongResponse>(peerAddress, ping, timeoutCts.Token);
            
            _logger.LogTrace("Peer at {Address} is healthy. Response: {Message}", peerAddress, response.Message);

            // Se o ping foi bem-sucedido, garantimos que o timestamp do par está atualizado.
            _nodeState.MergePeers(new[] { peerAddress });
        }
        catch (Exception ex) // Captura timeouts, erros de conexão, etc.
        {
            _logger.LogWarning("Peer at {Address} appears to be offline due to health check failure ({ErrorType}). Removing from peer list.", peerAddress, ex.GetType().Name);
            
            // Se o ping falhar, removemos o nó da nossa lista de pares.
            // A fofoca se encarregará de propagar essa remoção para o resto da rede.
            _nodeState.RemovePeer(peerAddress);
            _nodeState.PrintStatus();
        }
    }
}