using Galileu.Models;
using Galileu.Services;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;
using System.Security.Claims;
using System.Threading.Tasks; // Adicionado

namespace Galileu.API.Controllers;

public record PurchaseRequestDto(decimal Amount, string PaymentGatewayToken);
public record SellRequestDto(decimal Amount);
public record TransferRequestDto(string ToAddress, decimal Amount);

[ApiController]
[Route("api/[controller]")]
[Authorize]
public class TokenController : ControllerBase
{
    private readonly WalletService _walletService;
    private readonly RewardContractService _rewardContract;

    public TokenController(WalletService walletService, RewardContractService rewardContract)
    {
        _walletService = walletService;
        _rewardContract = rewardContract;
    }

    private string GetCurrentUserAddress() => User.FindFirstValue(ClaimTypes.NameIdentifier) 
                                              ?? throw new InvalidOperationException("User address not found in token.");

    [HttpPost("purchase")]
    public async Task<IActionResult> PurchaseTokens([FromBody] PurchaseRequestDto request)
    {
        var userAddress = GetCurrentUserAddress();
        bool paymentSucceeded = await MockPaymentGateway.ProcessPayment(request.PaymentGatewayToken, request.Amount);

        if (!paymentSucceeded)
        {
            return BadRequest(new { Message = "Pagamento falhou." });
        }

        await _walletService.CreateTransactionAsync(
            RewardContractService.SystemMintAddress, 
            userAddress, 
            request.Amount, 
            "User Purchase");
        var newBalance = await _walletService.GetBalanceAsync(userAddress);
        
        return Ok(new { Message = "Compra realizada com sucesso.", NewBalance = newBalance });
    }

    [HttpPost("sell")]
    public async Task<IActionResult> SellTokens([FromBody] SellRequestDto request)
    {
        var userAddress = GetCurrentUserAddress();
        
        // CORRIGIDO: Usa 'await' em vez de '.Result' para obter o saldo.
        var currentBalance = await _walletService.GetBalanceAsync(userAddress);
        if (currentBalance < request.Amount)
        {
            return BadRequest(new { Message = "Saldo insuficiente." });
        }

        await _rewardContract.BurnTokensAsync(userAddress, request.Amount, "User Sale");
        
        // CORRIGIDO: Usa 'await' para obter o novo saldo.
        var newBalance = await _walletService.GetBalanceAsync(userAddress);
        
        return Ok(new { Message = "Venda realizada com sucesso.", NewBalance = newBalance });
    }
    
    [HttpPost("transfer")]
    public async Task<IActionResult> TransferTokens([FromBody] TransferRequestDto request)
    {
        var fromAddress = GetCurrentUserAddress();
        
        try
        {
            await _walletService.CreateTransactionAsync(fromAddress, request.ToAddress, request.Amount, "User Transfer");
            
            // CORRIGIDO: Usa 'await' para obter o novo saldo.
            var newBalance = await _walletService.GetBalanceAsync(fromAddress);

            return Ok(new 
            {
                Message = "Transferência realizada com sucesso.",
                FromAddressNewBalance = newBalance 
            });
        }
        catch (InvalidOperationException ex)
        {
            return BadRequest(new { Message = ex.Message });
        }
    }
    
    [HttpGet("status")]
    [AllowAnonymous]
    public IActionResult GetTokenStatus()
    {
        // Este método está correto, pois GetTotalSupply é síncrono.
        return Ok(new { TotalSupply = _rewardContract.GetTotalSupply() });
    }

    [HttpGet("audit/ledger")]
    [Authorize(Roles = "Admin")]
    public async Task<IActionResult> GetLedger()
    {
        // CORRIGIDO: Proativamente tornado async para funcionar com um WalletService assíncrono.
        // Você precisará criar o método 'GetFullLedgerAsync' no seu WalletService.
        var ledger = await _walletService.GetFullLedgerAsync();
        return Ok(ledger);
    }
}

// Classe de simulação
public static class MockPaymentGateway
{
    public static Task<bool> ProcessPayment(string token, decimal amount)
    {
        Console.WriteLine($"Processando pagamento de {amount} com o token {token}...");
        return Task.FromResult(true);
    }
}