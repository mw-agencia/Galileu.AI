using Galileu.Models;
using Microsoft.AspNetCore.Mvc;

namespace Galileu.Controller;

public record NodeStatusDto(
    string Id, 
    string Address, 
    string? ParentAddress, 
    string? LeftChildAddress, 
    string? RightChildAddress
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
        // Usamos o DTO para formatar a resposta
        var status = new NodeStatusDto(
            _nodeState.Id,
            _nodeState.Address,
            _nodeState.ParentAddress,
            _nodeState.LeftChildAddress,
            _nodeState.RightChildAddress
        );
        
        return Ok(status);
    }
}