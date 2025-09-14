using Akka.Actor;

namespace Galileu.Services;

public record NodeInfo(
    IActorRef ActorRef,
    string WalletAddress,
    HashSet<string> Specializations // As áreas de expertise do nó
);