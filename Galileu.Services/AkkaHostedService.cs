using Akka.Actor;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.DependencyInjection; // Necessário para IServiceProvider

namespace Galileu.Services;

public class AkkaHostedService : IHostedService
{
    private readonly IServiceProvider _serviceProvider;
    private ActorSystem? _actorSystem;

    public AkkaHostedService(IServiceProvider serviceProvider)
    {
        _serviceProvider = serviceProvider;
    }

    public Task StartAsync(CancellationToken cancellationToken)
    {
        _actorSystem = ActorSystem.Create("Galileu-System");

        // Usamos um escopo para obter os serviços do container de DI
        using (var scope = _serviceProvider.CreateScope())
        {
            var registryService = scope.ServiceProvider.GetRequiredService<NodeRegistryService>();
            var rewardContractService = scope.ServiceProvider.GetRequiredService<RewardContractService>();
            var nodeState = scope.ServiceProvider.GetRequiredService<NodeState>(); // Obter o NodeState
            
            // --- CRIAÇÃO DOS ATORES ---
            _actorSystem.ActorOf(
                Props.Create(() => new SpecializedNodeActor(new[] { "Tradução" }, registryService, nodeState)),
                "node-translator");

            _actorSystem.ActorOf(
                Props.Create(() => new SpecializedNodeActor(new[] { "Análise de Sentimento" }, registryService, nodeState)),
                "node-sentiment");

            _actorSystem.ActorOf(
                Props.Create(() => new SpecializedNodeActor(new[] { "Geração de Código", "PLN" }, registryService, nodeState)),
                "node-coder-pln");

            var consensusCoordinator = _actorSystem.ActorOf(
                Props.Create(() => new ConsensusCoordinatorActor(rewardContractService, registryService)),
                "consensus-coordinator");

            var worker = _actorSystem.ActorOf(
                Props.Create(() => new WorkerActor(consensusCoordinator, registryService)), 
                "worker");
        }

        Console.WriteLine("[Akka] ActorSystem 'Galileu-System' iniciado e atores criados.");
        
        // Disponibiliza o ActorSystem para o resto da aplicação
        // Isso permite que controllers ou outros serviços possam enviar mensagens para os atores.
        var actorSystemSingleton = _serviceProvider.GetRequiredService<ActorSystemSingleton>();
        actorSystemSingleton.ActorSystem = _actorSystem;

        return Task.CompletedTask;
    }

    public async Task StopAsync(CancellationToken cancellationToken)
    {
        if (_actorSystem != null)
        {
            await _actorSystem.Terminate();
            Console.WriteLine("[Akka] ActorSystem 'Galileu-System' desligado.");
        }
    }
}

// Classe auxiliar para injetar o ActorSystem em outros lugares
public class ActorSystemSingleton
{
    public ActorSystem? ActorSystem { get; set; }
}