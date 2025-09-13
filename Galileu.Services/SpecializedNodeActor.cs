using Akka.Actor;
using Galileu.Models;

namespace Galileu.Services;

public class SpecializedNodeActor : ReceiveActor
{
    private readonly string _specialization;
    private readonly string _nodeId;

    public SpecializedNodeActor(string specialization)
    {
        _specialization = specialization;
        _nodeId = Self.Path.Name;

        // Etapa de Processamento da subtarefa
        Receive<Messages.ProcessSubtask>(msg =>
        {
            Console.WriteLine($"[{_nodeId}/{_specialization}] Recebeu subtarefa '{msg.Content}'. Processando...");
            // Simula o trabalho da RNN para gerar um fragmento
            var fragment = $"<{_specialization}_fragment:{msg.Content}_by_{_nodeId}>";
                
            // Envia o resultado de volta para o orquestrador (Coordenador de Consenso)
            Sender.Tell(new Messages.SubtaskResult(msg.TaskId, msg.SubtaskIndex, fragment, Self));
        });

        // Etapa de Validação Coletiva
        Receive<Messages.RequestValidation>(msg =>
        {
            Console.WriteLine($"[{_nodeId}] Recebeu pedido para validar o fragmento: '{msg.FragmentToValidate.Fragment}'");
                
            // Lógica de validação simplificada: aprova se o fragmento não for vazio.
            // Em um sistema real, isso envolveria análise semântica, verificação de formato, etc.
            bool isValid = !string.IsNullOrWhiteSpace(msg.FragmentToValidate.Fragment);

            Console.WriteLine($"[{_nodeId}] Votando... Válido: {isValid}");
            Sender.Tell(new Messages.ValidationVote(msg.TaskId, msg.FragmentToValidate.SubtaskIndex, isValid, Self));
        });
    }
}