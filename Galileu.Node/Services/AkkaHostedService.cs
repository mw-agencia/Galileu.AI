using Akka.Actor;
using Akka.Configuration;
using Galileu.Node.Models; // Adicionar este using
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.DependencyInjection;

namespace Services;

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
        // Obtém a porta do NodeState para configurar o Akka.Remote
        var nodeState = _serviceProvider.GetRequiredService<NodeState>();
        var akkaPort = new Uri(nodeState.Address).Port + 1000; 

        var config = ConfigurationFactory.ParseString(@$"
            akka {{
                actor {{
                    provider = remote
                }}
                remote {{
                    dot-netty.tcp {{
                        port = {akkaPort}
                        hostname = localhost
                    }}
                }}
            }}");

        _actorSystem = ActorSystem.Create("Galileu-System", config); // Cria o sistema com a config de rede
        
        // ... O resto da criação dos atores permanece o mesmo ...

        return Task.CompletedTask;
    }

    public Task StopAsync(CancellationToken cancellationToken)
    {
        throw new NotImplementedException();
    }
    // ...
}