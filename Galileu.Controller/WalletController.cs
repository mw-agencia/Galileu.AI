using System.Text;
using Galileu.Models;
using Galileu.Services;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;
using Microsoft.Extensions.Logging;

[ApiController]
[Route("api/[controller]")]
public class WalletController : ControllerBase
{
    private readonly WalletService _walletService;
    private readonly ILogger<WalletController> _logger;

    public WalletController(WalletService walletService, ILogger<WalletController> logger)
    {
        _walletService = walletService;
        _logger = logger;
    }

    [HttpPost("create")]
    public async Task<IActionResult> CreateWallet()
    {
        var (publicKey, privateKey) = CryptoUtils.GenerateKeyPair();
        
        // Armazena a nova carteira no banco de dados
        await _walletService.CreateWalletAsync(publicKey);

        // Retorna a chave privada APENAS UMA VEZ. Ã‰ responsabilidade do cliente guardÃ¡-la.
        return Ok(new { PublicKey = publicKey, PrivateKey = privateKey });
    }
    
    [HttpPost("balance")]
    public async Task<IActionResult> GetBalance([FromBody] WalletRequest request)
    {
        try
        {
            
            if (request == null || string.IsNullOrWhiteSpace(request.WalletAddress))
                return BadRequest(new { Message = "Endereço da carteira é obrigatório." });

            var balance = await _walletService.GetBalanceAsync(request.WalletAddress);
            if (balance == null)
                return NotFound(new { Message = "Carteira não encontrada." });

            return Ok(new
            {
                address = request.WalletAddress,
                balance
            });
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Erro ao processar saldo da carteira.");
            return StatusCode(500, new { Message = "Erro inesperado ao processar sua solicitação." });
        }
    }
    
    /// <summary>
    /// Consulta o histÃ³rico de transaÃ§Ãµes de uma carteira. O endereÃ§o deve ser codificado em Base64Url.
    /// </summary>
    [HttpPost("history")]
    public async Task<IActionResult> GetHistory([FromBody] WalletRequest request)
    {
        try
        {
            // Use diretamente, sem decodificação
            if (string.IsNullOrWhiteSpace(request.WalletAddress))
                return BadRequest(new { Message = "O endereço da carteira não pode ser vazio." });

            var history = await _walletService.GetHistoryAsync(request.WalletAddress);
            return Ok(history);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Erro ao processar histórico da carteira.");
            return StatusCode(500, new { Message = "Erro inesperado ao processar sua solicitação." });
        }
    }
    
    [HttpGet("encode/{value}")]
    [AllowAnonymous]
    public IActionResult EncodeForUrl(string value) => Ok(new { encoded = EncodeBase64Url(value) });
    
    private static string DecodeBase64Url(string input)
    {
        string output = input;
        output = output.Replace('-', '+'); // 62nd char of encoding
        output = output.Replace('_', '/'); // 63rd char of encoding
        switch (output.Length % 4) // Pad with trailing '='s
        {
            case 0: break; // No pad chars in this case
            case 2: output += "=="; break; // Two pad chars
            case 3: output += "="; break; // One pad char
            default: throw new FormatException("Formato Base64Url ilegal.");
        }
        var converted = Convert.FromBase64String(output);
        return Encoding.UTF8.GetString(converted);
    }
    
    private static string EncodeBase64Url(string input)
    {
        var bytes = Encoding.UTF8.GetBytes(input);
        return Convert.ToBase64String(bytes)
            .Replace('+', '-')
            .Replace('/', '_')
            .TrimEnd('=');
    }
}