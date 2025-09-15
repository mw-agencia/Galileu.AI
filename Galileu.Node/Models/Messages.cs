using System.Text.Json.Serialization;
using Akka.Actor;

namespace Galileu.Node.Models;

// ####################################################################
// ## MENSAGENS PARA A REDE P2P (WebSocket)                          ##
// ####################################################################

/// <summary>
/// A classe base para todas as mensagens da rede P2P.
/// A herança resolve todos os erros de "not assignable".
/// </summary>
[JsonDerivedType(typeof(JoinRequest), typeDiscriminator: "join_request")]
[JsonDerivedType(typeof(JoinResponse), typeDiscriminator: "join_response")]
[JsonDerivedType(typeof(ForwardJoinRequest), typeDiscriminator: "forward_join_request")]
[JsonDerivedType(typeof(PingRequest), typeDiscriminator: "ping_request")]
[JsonDerivedType(typeof(PongResponse), typeDiscriminator: "pong_response")]
[JsonDerivedType(typeof(GossipSyncRequest), typeDiscriminator: "gossip_sync_request")]
[JsonDerivedType(typeof(GossipSyncResponse), typeDiscriminator: "gossip_sync_response")]
public abstract record Message(Guid CorrelationId)
{
    
}

// --- Requisições P2P ---

public record JoinRequest(Guid CorrelationId, string NewNodeId, string NewNodeAddress) : Message(CorrelationId);
public record ForwardJoinRequest(Guid CorrelationId, JoinRequest OriginalRequest) : Message(CorrelationId);
public record PingRequest(Guid CorrelationId, string FromNodeId) : Message(CorrelationId);
public record JoinResponse(Guid CorrelationId, bool Success, string ParentNodeId, string Message) : Message(CorrelationId);
public record PongResponse(Guid CorrelationId, string Message) : Message(CorrelationId);


// ####################################################################
// ## MENSAGENS PARA O CONSENSO DE ATORES (Akka.NET)                 ##
// ####################################################################

// Um nó envia sua lista de pares conhecidos.
public record GossipSyncRequest(Guid CorrelationId, List<string> KnownPeers) : Message(CorrelationId);

// O outro nó responde com sua própria lista de pares.
public record GossipSyncResponse(Guid CorrelationId, List<string> KnownPeers) : Message(CorrelationId);
public record ProcessTaskRequest(string Prompt, Guid TaskId);
public record StartConsensusRound(Guid TaskId, string[] Subtasks, IActorRef Worker);
public record ProcessSubtask(Guid TaskId, int SubtaskIndex, string Content);
public record SubtaskResult(Guid TaskId, int SubtaskIndex, string Fragment, IActorRef Node);
public record RequestValidation(Guid TaskId, SubtaskResult FragmentToValidate);
public record ValidationVote(Guid TaskId, int SubtaskIndex, bool IsValid, IActorRef Voter);
public record ConsensusReached(Guid TaskId, List<SubtaskResult> ValidatedFragments);
public record ConsensusFailed(Guid TaskId, string Reason);

public class Messages 
{
    // Todos os records de mensagem aqui dentro
    public record ProcessTaskRequest(string Prompt, Guid TaskId);
    public record StartConsensusRound(Guid TaskId, string[] Subtasks, IActorRef Worker, List<object> SelectedNodes);
    public record ProcessSubtask(Guid TaskId, int SubtaskIndex, string Content);
    
    public record SubtaskResult(Guid TaskId, int SubtaskIndex, string Fragment, IActorRef Node);
    public record RequestValidation(Guid TaskId, SubtaskResult FragmentToValidate);
    public record ValidationVote(Guid TaskId, int SubtaskIndex, bool IsValid, IActorRef Voter);
    public record ConsensusReached(Guid TaskId, List<SubtaskResult> ValidatedFragments);
    public record ConsensusFailed(Guid TaskId, string Reason);
    // etc.
}