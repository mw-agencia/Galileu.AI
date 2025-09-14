// Galileu.API/Controllers/WalletController.cs

using Galileu.Models;
using Galileu.Services;
using Microsoft.AspNetCore.Mvc;

[ApiController]
[Route("api/[controller]")]
public class WalletController : ControllerBase
{
    private readonly WalletService _walletService;

    public WalletController(WalletService walletService)
    {
        _walletService = walletService;
    }

    [HttpPost("create")]
    public async Task<IActionResult> CreateWallet()
    {
        var (publicKey, privateKey) = CryptoUtils.GenerateKeyPair();
        
        // Armazena a nova carteira no banco de dados
        await _walletService.CreateWalletAsync(publicKey);

        // Retorna a chave privada APENAS UMA VEZ. É responsabilidade do cliente guardá-la.
        return Ok(new { PublicKey = publicKey, PrivateKey = privateKey });
    }
    
    [HttpGet("{walletAddress}/balance")]
    public async Task<IActionResult> GetBalance(string walletAddress)
    {
        var decodedAddress = System.Web.HttpUtility.UrlDecode(walletAddress);
        var balance = await _walletService.GetBalanceAsync(decodedAddress);
        return Ok(new { Address = decodedAddress, Balance = balance });
    }
    
    [HttpGet("{walletAddress}/history")]
    public async Task<IActionResult> GetHistory(string walletAddress)
    {
        var decodedAddress = System.Web.HttpUtility.UrlDecode(walletAddress);
        var history = await _walletService.GetHistoryAsync(decodedAddress);
        return Ok(history);
    }
}