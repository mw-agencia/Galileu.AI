using Galileu.Models;
using Galileu.Services;
using Microsoft.AspNetCore.Mvc;

namespace Galileu.Controller;

public record NodeStatusDto(
    string Id, 
    string Address, 
    int KnownPeersCount, // Quantidade de pares conhecidos
    List<string> KnownPeers // A lista de endereços dos pares conhecidos
);

[ApiController]
[Route("api/[controller]")] // A rota será /api/node
public class NodeController : ControllerBase
{
    private readonly NodeState _nodeState;

    // O NodeState é injetado automaticamente pelo container de DI
    public NodeController(NodeState nodeState)
    {
        _nodeState = nodeState;
    }

    /// <summary>
    /// Retorna o status atual deste nó na rede P2P.
    /// </summary>
    /// <returns>Informações sobre o nó, incluindo seus filhos e pai.</returns>
    [HttpGet("status")]
    [ProducesResponseType(typeof(NodeStatusDto), 200)]
    public IActionResult GetStatus()
    {
        // Usamos o DTO para formatar a resposta com as informações da rede Gossip
        var status = new NodeStatusDto(
            _nodeState.Id,
            _nodeState.Address,
            _nodeState.GetKnownPeers().Count, // Obtém a contagem de pares
            _nodeState.GetKnownPeers()        // Obtém a lista de pares
        );
        
        return Ok(status);
    }
}