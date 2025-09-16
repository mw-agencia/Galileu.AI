using System.IdentityModel.Tokens.Jwt;
using System.Security.Claims;
using System.Text;
using Galileu.Models;
using Galileu.Services;
using Microsoft.AspNetCore.Mvc;
using Microsoft.Extensions.Configuration;
using Microsoft.IdentityModel.Tokens;

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
    private readonly NodeRegistryService _nodeRegistry;
    private readonly NodeState _nodeState;
    private readonly IConfiguration _configuration;


    // O NodeState é injetado automaticamente pelo container de DI
    public NodeController(IConfiguration configuration,NodeState nodeState,NodeRegistryService nodeRegistry)
    {
        _nodeRegistry = nodeRegistry;
        _configuration  = configuration;
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
    [HttpPost("register")]
    public IActionResult RegisterNode([FromBody] NodeRegistrationRequest request)
    {
        var secretKey = _configuration["Jwt:SecretKey"]!;
        var issuer = _configuration["Jwt:Issuer"] ?? "GalileuAPI";
        var audience = _configuration["Jwt:NodeAudience"] ?? "GalileuNodes"; // Usa a audiência correta

        var securityKey = new SymmetricSecurityKey(Encoding.UTF8.GetBytes(secretKey));
        var credentials = new SigningCredentials(securityKey, SecurityAlgorithms.HmacSha256);

        var claims = new[]
        {
            new Claim(ClaimTypes.NameIdentifier, request.PublicWalletAddress),
            new Claim(ClaimTypes.Role, "node"),
            new Claim("node_address", request.NodeNetworkAddress)
        };

        var token = new JwtSecurityToken(
            issuer: issuer,
            audience: audience, // Usa a audiência correta
            claims: claims,
            expires: DateTime.UtcNow.AddYears(1),
            signingCredentials: credentials);

        var tokenString = new JwtSecurityTokenHandler().WriteToken(token);

        var knownPeers = _nodeRegistry.GetAllNodeAddresses().Where(addr => addr != request.NodeNetworkAddress);

        return Ok(new 
        { 
            NodeJwt = tokenString,
            InitialPeers = knownPeers 
        });
    }
}