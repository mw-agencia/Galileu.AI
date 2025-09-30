using Akka.Actor;

namespace Galileu.Services;

public record NodeInfo(
    IActorRef ActorRef,
    string WalletAddress,
    string NetworkAddress,
    HashSet<string> Specializations // As áreas de expertise do nó
);