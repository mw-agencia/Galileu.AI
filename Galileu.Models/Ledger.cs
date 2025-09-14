using System.Security.Cryptography;
using System.Text;

namespace Galileu.Models;

// Representa uma única transação no nosso sistema
public record Transaction(
    Guid Id,
    DateTime Timestamp,
    string FromAddress, // "SYSTEM" para recompensas, ou o endereço de um usuário
    string ToAddress,
    decimal Amount,
    string? Notes,
    string Hash // Hash da transação para garantir integridade
);

public static class CryptoUtils
{
    // Gera um par de chaves para um novo usuário (carteira)
    public static (string publicKey, string privateKey) GenerateKeyPair()
    {
        using var rsa = RSA.Create();
        return (
            publicKey: Convert.ToBase64String(rsa.ExportRSAPublicKey()),
            privateKey: Convert.ToBase64String(rsa.ExportRSAPrivateKey())
        );
    }

    // Cria um hash para garantir a imutabilidade da transação
    public static string CreateTransactionHash(Guid id, DateTime timestamp, string from, string to, decimal amount)
    {
        using var sha256 = SHA256.Create();
        var data = $"{id}{timestamp:O}{from}{to}{amount}";
        var hashBytes = sha256.ComputeHash(Encoding.UTF8.GetBytes(data));
        return Convert.ToBase64String(hashBytes);
    }
}