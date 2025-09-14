using Galileu.Models;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;
using MongoDB.Driver;
using MongoDB.Driver.Linq;

namespace Galileu.Services;

public class WalletService
{
    private readonly IMongoCollection<TransactionDocument> _transactions;
    private readonly IMongoCollection<WalletDocument> _wallets;
    private readonly ILogger<WalletService> _logger;

    public WalletService(IOptions<MongoDbSettings> mongoSettings, ILogger<WalletService> logger)
    {
        _logger = logger;
        var client = new MongoClient(mongoSettings.Value.ConnectionString);
        var database = client.GetDatabase(mongoSettings.Value.DatabaseName);

        _transactions = database.GetCollection<TransactionDocument>("Transactions");
        _wallets = database.GetCollection<WalletDocument>("Wallets");
    }
    
    // Método para criar e armazenar uma nova carteira
    public async Task CreateWalletAsync(string address)
    {
        var wallet = new WalletDocument
        {
            Address = address,
            CreatedAt = DateTime.UtcNow
        };
        await _wallets.InsertOneAsync(wallet);
        _logger.LogInformation("Nova carteira criada e armazenada no DB para o endereço: {Address}", address);
    }
    
    public async Task CreateTransactionAsync(string fromAddress, string toAddress, decimal amount, string? notes = null)
    {
        if (fromAddress != RewardContractService.SystemMintAddress)
        {
            var balance = await GetBalanceAsync(fromAddress);
            if (balance < amount)
            {
                throw new InvalidOperationException("Saldo insuficiente.");
            }
        }

        var transaction = new TransactionDocument
        {
            Timestamp = DateTime.UtcNow,
            FromAddress = fromAddress,
            ToAddress = toAddress,
            Amount = amount,
            Notes = notes,
            Hash = CryptoUtils.CreateTransactionHash(Guid.NewGuid(), DateTime.UtcNow, fromAddress, toAddress, amount)
        };

        await _transactions.InsertOneAsync(transaction);
        _logger.LogInformation("Transação de {Amount} de {From} para {To} registrada no MongoDB.", amount, fromAddress, toAddress);
    }
    
    public async Task<decimal> GetBalanceAsync(string walletAddress)
    {
        // Usamos o poderoso Aggregation Framework do MongoDB para calcular o saldo
        var receivedTask = _transactions.AsQueryable()
            .Where(t => t.ToAddress == walletAddress)
            .SumAsync(t => t.Amount);

        var sentTask = _transactions.AsQueryable()
            .Where(t => t.FromAddress == walletAddress)
            .SumAsync(t => t.Amount);
            
        await Task.WhenAll(receivedTask, sentTask);

        return receivedTask.Result - sentTask.Result;
    }

    public async Task<IEnumerable<TransactionDocument>> GetHistoryAsync(string walletAddress)
    {
        var filter = Builders<TransactionDocument>.Filter.Or(
            Builders<TransactionDocument>.Filter.Eq(t => t.FromAddress, walletAddress),
            Builders<TransactionDocument>.Filter.Eq(t => t.ToAddress, walletAddress)
        );

        return await _transactions.Find(filter)
            .SortByDescending(t => t.Timestamp)
            .ToListAsync();
    }
    
    public async Task<IEnumerable<TransactionDocument>> GetFullLedgerAsync()
    {
        // Retorna todas as transações, ordenadas pela mais recente primeiro
        return await _transactions.Find(_ => true)
            .SortByDescending(t => t.Timestamp)
            .ToListAsync();
    }
}