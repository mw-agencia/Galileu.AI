using Akka.Actor;
using Galileu.Models; // Onde CryptoUtils está

namespace Galileu.Services;

public class SpecializedNodeActor : ReceiveActor
{
    private readonly NodeRegistryService _registryService;
    private readonly IEnumerable<string> _specializations;
    private readonly string _nodeId;
    private readonly string _myWalletAddress;
    private readonly NodeState _nodeState;

    public SpecializedNodeActor(IEnumerable<string> specializations,
        NodeRegistryService registryService,
        NodeState nodeState)
    {
        _registryService = registryService;
        _specializations = specializations;
        _nodeState = nodeState;
        _nodeId = Self.Path.Name;
        
        // Corrigido: Gera a carteira e atribui o endereço
        var (publicKey, _) = CryptoUtils.GenerateKeyPair();
        _myWalletAddress = publicKey;

        // Lógica de recebimento de mensagens
        
        // Etapa de Processamento da subtarefa
        Receive<Messages.ProcessSubtask>(msg =>
        {
            // Corrigido: Exibe as especializações de forma legível
            var specsString = string.Join(", ", _specializations);
            Console.WriteLine($"[{_nodeId}/{specsString}] Recebeu subtarefa '{msg.Content}'. Processando...");
            
            // Simula o trabalho da RNN para gerar um fragmento
            var fragment = $"<{specsString}_fragment:{msg.Content}_by_{_nodeId}>";
                
            // Envia o resultado de volta para o orquestrador (Coordenador de Consenso)
            Sender.Tell(new Messages.SubtaskResult(msg.TaskId, msg.SubtaskIndex, fragment, Self));
        });

        // Etapa de Validação Coletiva
        Receive<Messages.RequestValidation>(msg =>
        {
            Console.WriteLine($"[{_nodeId}] Recebeu pedido para validar o fragmento: '{msg.FragmentToValidate.Fragment}'");
                
            // Lógica de validação simplificada
            bool isValid = !string.IsNullOrWhiteSpace(msg.FragmentToValidate.Fragment);

            Console.WriteLine($"[{_nodeId}] Votando... Válido: {isValid}");
            Sender.Tell(new Messages.ValidationVote(msg.TaskId, msg.FragmentToValidate.SubtaskIndex, isValid, Self));
        });
    }

    /// <summary>
    /// Método do ciclo de vida do ator, chamado antes de começar a processar mensagens.
    /// É o lugar ideal para registrar o nó.
    /// </summary>
    protected override void PreStart()
    {
        // Agora passa o endereço de rede do NodeState durante o registro
        _registryService.RegisterNode(Self, _myWalletAddress, _nodeState.Address, _specializations);
        base.PreStart();
    }

    /// <summary>
    /// Método do ciclo de vida do ator, chamado após o ator ser parado.
    /// Ideal para limpar recursos, como remover o registro.
    /// </summary>
    protected override void PostStop()
    {
        _registryService.UnregisterNode(Self);
        base.PostStop();
    }
}