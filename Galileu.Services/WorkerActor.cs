using Akka.Actor;
using Galileu.Models;

namespace Galileu.Services;

public class WorkerActor : ReceiveActor
{
    private readonly IActorRef _consensusCoordinator;

    public WorkerActor(IActorRef consensusCoordinator)
    {
        _consensusCoordinator = consensusCoordinator;

        Receive<Messages.ProcessTaskRequest>(msg =>
        {
            Console.WriteLine($"[Worker] Recebeu nova solicitação do usuário: '{msg.Prompt}'");
                
            // 1. DIVISÃO DA TAREFA
            // Lógica de fragmentação simples: divide o prompt por espaços.
            // Um sistema real usaria PLN para uma divisão mais inteligente.
            var subtasks = msg.Prompt.Split(' ');
            Console.WriteLine($"[Worker] Tarefa dividida em {subtasks.Length} subtarefas.");

            // Inicia a rodada de consenso enviando as subtarefas para o Coordenador
            _consensusCoordinator.Tell(new Messages.StartConsensusRound(msg.TaskId, subtasks, Self));
        });

        Receive<Messages.ConsensusReached>(msg =>
        {
            Console.WriteLine($"[Worker] Consenso alcançado para a Tarefa ID: {msg.TaskId}!");
                
            // 4. AGREGAÇÃO (final)
            var finalResponse = string.Join(" ", msg.ValidatedFragments.Select(f => f.Fragment));
                
            Console.WriteLine("\n================== RESPOSTA FINAL GERADA ==================");
            Console.WriteLine(finalResponse);
            Console.WriteLine("=========================================================\n");
        });
            
        Receive<Messages.ConsensusFailed>(msg =>
        {
            Console.WriteLine($"\n[Worker] FALHA NO CONSENSO para a Tarefa ID: {msg.TaskId}. Motivo: {msg.Reason}\n");
        });
    }
}