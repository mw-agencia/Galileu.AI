using Microsoft.Extensions.Logging;

namespace Galileu.Services;

// Este é o nosso "Smart Contract". Uma autoridade central e confiável.
public class RewardContractService
{
    private readonly WalletService _walletService;
    private readonly ILogger<RewardContractService> _logger;
    private decimal _totalSupply = 0; // Oferta total de tokens em circulação
    private readonly object _lock = new();

    public const string SystemMintAddress = "SYSTEM_MINT";
    public const string SystemBurnAddress = "SYSTEM_BURN";

    public RewardContractService(WalletService walletService, ILogger<RewardContractService> logger)
    {
        _walletService = walletService;
        _logger = logger;
    }

    public decimal GetTotalSupply() => _totalSupply;

    /// <summary>
    /// Registra e valida uma contribuição, disparando a recompensa se for válida.
    /// </summary>
    public async Task ProcessContribution(Guid taskId, string userWalletAddress, decimal contributionScore)
    {
        var rewardAmount = CalculateReward(contributionScore);
        if (rewardAmount <= 0) return;

        // Cria a transação de "mint"
        await _walletService.CreateTransactionAsync(SystemMintAddress, userWalletAddress, rewardAmount, $"Reward for TaskId: {taskId}");

        // Atualiza a oferta total de forma thread-safe
        lock (_lock)
        {
            _totalSupply += rewardAmount;
        }
        
        _logger.LogInformation("MINTED: {Amount} GLU. New total supply: {_totalSupply}", rewardAmount, _totalSupply);
    }
    
    public async Task BurnTokensAsync(string fromAddress, decimal amount, string reason)
    {
        // Cria uma transação para o "endereço de queima"
        await _walletService.CreateTransactionAsync(fromAddress, SystemBurnAddress, amount, $"Burn: {reason}");
        
        // Atualiza a oferta total de forma thread-safe
        lock (_lock)
        {
            _totalSupply -= amount;
        }

        _logger.LogInformation("BURNED: {Amount} GLU. New total supply: {_totalSupply}", amount, _totalSupply);
    }
    
    public async Task MintTokensAsync(string toAddress, decimal amount, string reason)
    {
        // Cria a transação de "mint"
        await _walletService.CreateTransactionAsync(SystemMintAddress, toAddress, amount, $"GLU: {reason}");

        // Atualiza a oferta total de forma thread-safe
        lock (_lock)
        {
            _totalSupply += amount;
        }
        
        _logger.LogInformation("MINTED: {Amount} GLU for {ToAddress}. New total supply: {_totalSupply}", amount, toAddress, _totalSupply);
    }

    private decimal CalculateReward(decimal score)
    {
        return score * 10m;
    }
}