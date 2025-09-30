using System.Text.RegularExpressions;
using Akka.Actor;
using Galileu.Models;

namespace Galileu.Services;

public class WorkerActor : ReceiveActor
{
    private readonly IActorRef _consensusCoordinator;
    private readonly NodeRegistryService _nodeRegistry;
    private readonly Dictionary<Guid, IActorRef> _requesters = new();

    public WorkerActor(IActorRef consensusCoordinator, NodeRegistryService nodeRegistry)
    {
        _consensusCoordinator = consensusCoordinator;
        _nodeRegistry = nodeRegistry;
        Ready();
    }
    
    private void Ready()
    {
        // 1. RECEBER A TAREFA DO USUÁRIO (ou de um API Controller)
        Receive<Messages.ProcessTaskRequest>(msg =>
        {
            Console.WriteLine($"[Worker] Recebeu nova solicitação: '{msg.Prompt}' (TaskId: {msg.TaskId})");
            _requesters[msg.TaskId] = Sender; // Salva quem pediu a tarefa

            // 2. INTERPRETAÇÃO DO INPUT
            var requiredSpecializations = InterpretPrompt(msg.Prompt);
            Console.WriteLine($"[Worker] Especializações identificadas: {string.Join(", ", requiredSpecializations)}");

            // 3. SELEÇÃO E ATIVAÇÃO DOS NÓS
            var selectedNodes = SelectNodes(requiredSpecializations);
            if (!selectedNodes.Any())
            {
                Console.WriteLine("[Worker] Nenhum nó qualificado encontrado. Abortando tarefa.");
                Sender.Tell(new Messages.ConsensusFailed(msg.TaskId, "Nenhum nó qualificado disponível."));
                return;
            }

            Console.WriteLine($"[Worker] Nós selecionados para a tarefa: {string.Join(", ", selectedNodes.Select<NodeInfo, object>(n => n.ActorRef.Path.Name))}");

            // 4. DIVISÃO DA TAREFA E INÍCIO DO CONSENSO
            // A lógica de divisão pode ser mais inteligente, mas aqui vamos manter simples
            var subtasks = msg.Prompt.Split(' '); 
            _consensusCoordinator.Tell(new Messages.StartConsensusRound(msg.TaskId, subtasks, Self, selectedNodes.Select<NodeInfo, object>(n => n.ActorRef).ToList()));
        });
        
        // 5. AGREGAÇÃO E ENTREGA DA RESPOSTA
        Receive<Messages.ConsensusReached>(msg =>
        {
            Console.WriteLine($"[Worker] Consenso alcançado para a Tarefa ID: {msg.TaskId}!");
            
            // Agrega os fragmentos em uma única resposta
            var finalResponse = string.Join(" ", msg.ValidatedFragments.Select(f => f.Fragment));
            
            // Tenta encontrar quem pediu a tarefa e envia a resposta de volta
            if (_requesters.TryGetValue(msg.TaskId, out var requester))
            {
                // Aqui, a resposta poderia ser um objeto mais complexo, não apenas uma string
                requester.Tell(finalResponse); 
                _requesters.Remove(msg.TaskId);
            }
        });
        
        Receive<Messages.ConsensusFailed>(msg =>
        {
            Console.WriteLine($"\n[Worker] FALHA NO CONSENSO para a Tarefa ID: {msg.TaskId}. Motivo: {msg.Reason}\n");
            if (_requesters.TryGetValue(msg.TaskId, out var requester))
            {
                requester.Tell($"Falha ao processar a tarefa: {msg.Reason}");
                _requesters.Remove(msg.TaskId);
            }
        });
    }
    
    /// <summary>
    /// Responsabilidade 1: Interpreta o prompt para identificar as especializações necessárias.
    /// </summary>
    private List<string> InterpretPrompt(string prompt)
    {
        var specializations = new List<string>();
        var lowerPrompt = prompt.ToLower();

        if (Regex.IsMatch(lowerPrompt, @"\b(traduza|translate|alemão|inglês|espanhol)\b"))
            specializations.Add("Tradução");
            
        if (Regex.IsMatch(lowerPrompt, @"\b(sentimento|opinião|feliz|triste|positivo|negativo)\b"))
            specializations.Add("Análise de Sentimento");

        if (Regex.IsMatch(lowerPrompt, @"\b(código|code|função|function|classe|class|c#)\b"))
            specializations.Add("Geração de Código");

        return specializations;
    }
    
    /// <summary>
    /// Responsabilidade 2 & 3: Consulta o registro e seleciona os nós.
    /// </summary>
    private List<NodeInfo> SelectNodes(List<string> requiredSpecializations)
    {
        if (!requiredSpecializations.Any())
        {
            // Se nenhuma especialização for encontrada, pega 3 nós de propósito geral
            return _nodeRegistry.GetGeneralPurposeNodes(3).ToList();
        }

        var selectedNodes = new HashSet<NodeInfo>();
        foreach (var spec in requiredSpecializations)
        {
            var foundNodes = _nodeRegistry.FindNodesBySpecialization(spec);
            foreach (var node in foundNodes)
            {
                selectedNodes.Add(node);
            }
        }
        return selectedNodes.ToList();
    }
}