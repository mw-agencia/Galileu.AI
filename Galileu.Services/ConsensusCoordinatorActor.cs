// Galileu.Services/ConsensusCoordinatorActor.cs
using Akka.Actor;
using Galileu.Models;

namespace Galileu.Services;

/// <summary>
/// O back-end do worker de sincronização. Orquestra o Protocolo de Consenso.
/// </summary>
public class ConsensusCoordinatorActor : ReceiveActor
{
    private readonly RewardContractService _rewardContract;
    private readonly NodeRegistryService _nodeRegistry;
    
    // --- Estado da Rodada de Consenso Atual ---
    private Guid _currentTaskId;
    private IActorRef? _worker;
    private int _totalSubtasks;
    private List<IActorRef> _participatingNodes = new();
    
    // Dicionários para rastrear o progresso
    private readonly Dictionary<int, Messages.SubtaskResult> _results = new();
    private readonly Dictionary<int, List<Messages.ValidationVote>> _votes = new();

    public ConsensusCoordinatorActor(RewardContractService rewardContract, NodeRegistryService nodeRegistry)
    {
        _rewardContract = rewardContract;
        _nodeRegistry = nodeRegistry;
        
        // O ator começa no estado 'Idle' (Ocioso)
        Become(Idle);
    }

    /// <summary>
    /// Estado 1: Ocioso. Aguardando uma nova tarefa de sincronização.
    /// </summary>
    private void Idle()
    {
        Receive<Messages.StartConsensusRound>(msg =>
        {
            // --- 1. INÍCIO DA SINCRONIZAÇÃO E DIVISÃO DA TAREFA ---
            Console.WriteLine($"[SyncCoordinator] Iniciando consenso para Tarefa {msg.TaskId}. Nós participantes: {msg.SelectedNodes.Count}");
            
            // Configura o estado para a nova rodada
            _currentTaskId = msg.TaskId;
            _worker = msg.Worker;
            _totalSubtasks = msg.Subtasks.Length;
            _participatingNodes = msg.SelectedNodes.Cast<IActorRef>().ToList();
            _results.Clear();
            _votes.Clear();

            if (!_participatingNodes.Any() || _totalSubtasks == 0)
            {
                _worker.Tell(new Messages.ConsensusFailed(_currentTaskId, "Nenhum nó ou subtarefa para processar."));
                return; // Volta a ficar ocioso
            }

            // Distribui as subtarefas para os nós selecionados em um padrão de rodízio (round-robin)
            for (int i = 0; i < _totalSubtasks; i++)
            {
                var subtask = new Messages.ProcessSubtask(_currentTaskId, i, msg.Subtasks[i]);
                var targetNode = _participatingNodes[i % _participatingNodes.Count];
                targetNode.Tell(subtask); 
            }
            
            // Muda para o próximo estado, aguardando os resultados
            Become(WaitingForResults);
        });
    }

    /// <summary>
    /// Estado 2: Aguardando os fragmentos de resposta de todos os nós.
    /// </summary>
    private void WaitingForResults()
    {
        Receive<Messages.SubtaskResult>(msg =>
        {
            if (msg.TaskId != _currentTaskId) return; // Ignora mensagem de tarefa antiga
            
            Console.WriteLine($"[SyncCoordinator] Recebeu fragmento {msg.SubtaskIndex} do nó {msg.Node.Path.Name}");
            _results[msg.SubtaskIndex] = msg;

            // --- 2. SINCRONIZAÇÃO TEMPORAL ---
            // A transição de estado só ocorre quando todas as partes chegaram.
            if (_results.Count == _totalSubtasks)
            {
                Console.WriteLine("[SyncCoordinator] Todos os fragmentos recebidos. Iniciando fase de validação coletiva.");
                StartValidationPhase();
                Become(WaitingForVotes);
            }
        });
    }

    /// <summary>
    /// Inicia a fase de validação coletiva.
    /// </summary>
    private void StartValidationPhase()
    {
        // --- 3. VALIDAÇÃO COLETIVA ---
        // Pede para que todos os nós participantes validem cada fragmento recebido.
        foreach (var result in _results.Values)
        {
            var validationMsg = new Messages.RequestValidation(_currentTaskId, result);
            foreach (var node in _participatingNodes)
            {
                 node.Tell(validationMsg, Self);
            }
        }
    }

    /// <summary>
    /// Estado 3: Aguardando os votos de validação de todos os nós para todos os fragmentos.
    /// </summary>
    private void WaitingForVotes()
    {
        Receive<Messages.ValidationVote>(msg =>
        {
            if (msg.TaskId != _currentTaskId) return;

            if (!_votes.ContainsKey(msg.SubtaskIndex))
            {
                _votes[msg.SubtaskIndex] = new List<Messages.ValidationVote>();
            }
            _votes[msg.SubtaskIndex].Add(msg);

            // Verifica se o ciclo de votação está completo
            var totalExpectedVotes = _totalSubtasks * _participatingNodes.Count;
            var currentTotalVotes = _votes.Values.Sum(v => v.Count);
            
            if (currentTotalVotes >= totalExpectedVotes)
            {
                AggregateAndFinalize();
            }
        });
    }
    
    /// <summary>
    /// Agrega os resultados, finaliza o consenso e dispara as recompensas.
    /// </summary>
    private void AggregateAndFinalize()
    {
        Console.WriteLine("[SyncCoordinator] Votação concluída. Agregando resultados...");
        
        // --- 4. AGREGAÇÃO ---
        var validatedFragments = new List<Messages.SubtaskResult>();
        foreach (var subtaskIndex in _results.Keys)
        {
            var votesForFragment = _votes[subtaskIndex];
            var approvalCount = votesForFragment.Count(v => v.IsValid);
            
            // Regra de consenso: mais de 50% dos nós devem aprovar
            if (approvalCount > _participatingNodes.Count / 2)
            {
                Console.WriteLine($"[SyncCoordinator] Fragmento {subtaskIndex} APROVADO.");
                validatedFragments.Add(_results[subtaskIndex]);
            }
            else
            {
                Console.WriteLine($"[SyncCoordinator] Fragmento {subtaskIndex} REJEITADO.");
            }
        }

        // Verifica se todas as partes da resposta foram aprovadas
        if (validatedFragments.Count == _totalSubtasks)
        {
            var orderedFragments = validatedFragments.OrderBy(f => f.SubtaskIndex).ToList();
            _worker.Tell(new Messages.ConsensusReached(_currentTaskId, orderedFragments));

            // Dispara a distribuição de recompensas em uma Task separada para não bloquear o ator
            Task.Run(() => DistributeRewards(validatedFragments));
        }
        else
        {
            _worker.Tell(new Messages.ConsensusFailed(_currentTaskId, "Um ou mais fragmentos da resposta foram rejeitados no consenso."));
        }
        
        // Retorna ao estado ocioso, pronto para a próxima tarefa
        Become(Idle); 
    }

    private async Task DistributeRewards(List<Messages.SubtaskResult> validFragments)
    {
        Console.WriteLine("[SyncCoordinator] Iniciando distribuição de recompensas de tokens...");
        foreach (var fragment in validFragments)
        {
            var walletAddress = _nodeRegistry.GetWalletAddress(fragment.Node);
            if (walletAddress != null)
            {
                // A "qualidade" da contribuição define a recompensa. Aqui, usamos 1.0 como base.
                decimal contributionScore = 1.0m; 
                await _rewardContract.ProcessContribution(_currentTaskId, walletAddress, contributionScore);
            }
            else
            {
                Console.WriteLine($"[SyncCoordinator] AVISO: Não foi possível encontrar a carteira para o nó {fragment.Node.Path.Name}. Nenhuma recompensa será enviada.");
            }
        }
        Console.WriteLine("[SyncCoordinator] Distribuição de recompensas concluída.");
    }
}