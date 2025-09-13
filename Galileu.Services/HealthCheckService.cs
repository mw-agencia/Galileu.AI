using Galileu.Models;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;

namespace Galileu.Services;

public class HealthCheckService : BackgroundService
{
    private readonly NodeState _nodeState;
    private readonly NodeClient _nodeClient;
    private readonly ILogger<HealthCheckService> _logger;
    private readonly TimeSpan _checkInterval = TimeSpan.FromSeconds(15);

    public HealthCheckService(NodeState nodeState, NodeClient nodeClient, ILogger<HealthCheckService> logger)
    {
        _nodeState = nodeState;
        _nodeClient = nodeClient;
        _logger = logger;
    }

    protected override async Task ExecuteAsync(CancellationToken stoppingToken)
    {
        while (!stoppingToken.IsCancellationRequested)
        {
            await Task.Delay(_checkInterval, stoppingToken);
            await ValidateChildNode(_nodeState.LeftChildAddress, isLeft: true, stoppingToken);
            await ValidateChildNode(_nodeState.RightChildAddress, isLeft: false, stoppingToken);
        }
    }

    private async Task ValidateChildNode(string? childAddress, bool isLeft, CancellationToken token)
    {
        if (childAddress == null) return;
        _logger.LogInformation($"Pinging {(isLeft ? "left" : "right")} child at {childAddress}");
        
        try
        {
            var ping = new PingRequest(Guid.NewGuid(), _nodeState.Id);
            var response = await _nodeClient.SendRequestAsync<PongResponse>(childAddress, ping, token);
            _logger.LogInformation($"Child at {childAddress} is healthy. Response: {response.Message}");
        }
        catch (Exception ex) // Captura qualquer erro de comunicação (timeout, conexão recusada, etc.)
        {
            _logger.LogWarning(ex, $"Child node at {childAddress} appears to be offline. Removing from tree.");
            if (isLeft)
            {
                _nodeState.LeftChildAddress = null;
            }
            else
            {
                _nodeState.RightChildAddress = null;
            }
            _nodeState.PrintStatus();
        }
    }
}