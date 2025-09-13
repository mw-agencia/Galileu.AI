using Akka.Actor;
using Galileu.Models; // Contém a classe 'Messages'

namespace Galileu.Services;

public class ConsensusCoordinatorActor : ReceiveActor
{
    // Estado para gerenciar uma rodada de consenso
    private Guid _currentTaskId;
    private IActorRef _worker;
    // CORRIGIDO: Usar Messages.SubtaskResult
    private Dictionary<int, Messages.SubtaskResult> _results = new();
    // CORRIGIDO: Usar Messages.ValidationVote
    private Dictionary<int, List<Messages.ValidationVote>> _votes = new();
    private int _totalSubtasks;
    
    public ConsensusCoordinatorActor()
    {
        Become(Idle); // Começa no estado ocioso
    }

    // Estado Ocioso: esperando por uma nova tarefa
    private void Idle()
    {
        // CORRIGIDO: Usar Messages.StartConsensusRound
        Receive<Messages.StartConsensusRound>(msg =>
        {
            Console.WriteLine($"[Coordenador] Iniciando rodada de consenso para a Tarefa ID: {msg.TaskId}");
            
            // Configura o estado para a nova rodada
            _currentTaskId = msg.TaskId;
            _worker = msg.Worker;
            _totalSubtasks = msg.Subtasks.Length;
            _results.Clear();
            _votes.Clear();

            var availableNodes = Context.ActorSelection("/user/nodes/*");
            for (int i = 0; i < msg.Subtasks.Length; i++)
            {
                // CORRIGIDO: Usar Messages.ProcessSubtask
                var subtask = new Messages.ProcessSubtask(_currentTaskId, i, msg.Subtasks[i]);
                availableNodes.Tell(subtask); 
            }
            
            Become(WaitingForResults); // Muda para o estado de espera por resultados
        });
    }

    // Estado de Espera por Resultados dos Nós
    private void WaitingForResults()
    {
        // CORRIGIDO: Usar Messages.SubtaskResult
        Receive<Messages.SubtaskResult>(msg =>
        {
            if (msg.TaskId != _currentTaskId) return; 
            
            Console.WriteLine($"[Coordenador] Recebeu resultado para subtarefa {msg.SubtaskIndex}: '{msg.Fragment}'");
            _results[msg.SubtaskIndex] = msg;

            if (_results.Count == _totalSubtasks)
            {
                Console.WriteLine("[Coordenador] Todos os fragmentos recebidos. Iniciando fase de validação coletiva.");
                StartValidationPhase();
                Become(WaitingForVotes);
            }
        });
    }

    private void StartValidationPhase()
    {
        var availableNodes = Context.ActorSelection("/user/nodes/*");
        foreach (var result in _results.Values)
        {
            // CORRIGIDO: Usar Messages.RequestValidation
            var validationMsg = new Messages.RequestValidation(_currentTaskId, result);
            availableNodes.Tell(validationMsg, Self);
        }
    }

    // Estado de Espera por Votos de Validação
    private void WaitingForVotes()
    {
        // CORRIGIDO: Usar Messages.ValidationVote
        Receive<Messages.ValidationVote>(msg =>
        {
            if (msg.TaskId != _currentTaskId) return;

            if (!_votes.ContainsKey(msg.SubtaskIndex))
            {
                // CORRIGIDO: Usar Messages.ValidationVote
                _votes[msg.SubtaskIndex] = new List<Messages.ValidationVote>();
            }
            _votes[msg.SubtaskIndex].Add(msg);

            var totalExpectedVotes = _totalSubtasks * _totalSubtasks; 
            if (_votes.Values.Sum(v => v.Count) >= totalExpectedVotes)
            {
                AggregateAndFinalize();
            }
        });
    }
    
    private void AggregateAndFinalize()
    {
        Console.WriteLine("[Coordenador] Todos os votos recebidos. Agregando resultados...");
        
        // CORRIGIDO: Usar Messages.SubtaskResult
        var validatedFragments = new List<Messages.SubtaskResult>();
        foreach (var subtaskIndex in _results.Keys)
        {
            var votesForFragment = _votes[subtaskIndex];
            var approvalCount = votesForFragment.Count(v => v.IsValid);
            if (approvalCount > votesForFragment.Count / 2)
            {
                Console.WriteLine($"[Coordenador] Fragmento {subtaskIndex} APROVADO.");
                validatedFragments.Add(_results[subtaskIndex]);
            }
            else
            {
                Console.WriteLine($"[Coordenador] Fragmento {subtaskIndex} REJEITADO.");
            }
        }

        if (validatedFragments.Count == _totalSubtasks)
        {
            var orderedFragments = validatedFragments.OrderBy(f => f.SubtaskIndex).ToList();
            _worker.Tell(new Messages.ConsensusReached(_currentTaskId, orderedFragments));
        }
        else
        {
            _worker.Tell(new Messages.ConsensusFailed(_currentTaskId, "Nem todos os fragmentos foram validados com sucesso."));
        }
        
        Become(Idle); 
    }
}