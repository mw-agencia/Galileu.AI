// Localização: Galileu.Services/JwtValidatorService.cs

using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.Logging;
using Microsoft.IdentityModel.Tokens;
using System.IdentityModel.Tokens.Jwt;
using System.Security.Claims;
using System.Text;

namespace Galileu.Services;

/// <summary>
/// Um serviço dedicado para validar tokens JWT da rede de nós Galileu.
/// </summary>
public class JwtValidatorService
{
    private readonly ILogger<JwtValidatorService> _logger;
    private readonly TokenValidationParameters _tokenValidationParameters;

    public JwtValidatorService(IConfiguration configuration, ILogger<JwtValidatorService> logger)
    {
        _logger = logger;

        // Lê a chave secreta, o emissor e a audiência do appsettings.json
        var secretKey = configuration["Jwt:SecretKey"];
        var validIssuer = configuration["Jwt:Issuer"] ?? "GalileuAPI";
        var validAudience = configuration["Jwt:NodeAudience"] ?? "GalileuNodes";

        if (string.IsNullOrEmpty(secretKey))
        {
            _logger.LogCritical("A chave secreta JWT (Jwt:SecretKey) não está configurada no appsettings.json!");
            throw new InvalidOperationException("A chave secreta JWT não pode ser nula ou vazia.");
        }

        // Configura os parâmetros que serão usados para validar qualquer token
        _tokenValidationParameters = new TokenValidationParameters
        {
            // Valida a chave de assinatura
            ValidateIssuerSigningKey = true,
            IssuerSigningKey = new SymmetricSecurityKey(Encoding.UTF8.GetBytes(secretKey)),

            // Valida o emissor (quem criou o token)
            ValidateIssuer = true,
            ValidIssuer = validIssuer,

            // Valida a audiência (para quem o token foi destinado)
            ValidateAudience = true,
            ValidAudience = validAudience,

            // Valida o tempo de vida do token (não expirado)
            ValidateLifetime = true,

            // Permite uma pequena variação de tempo para evitar problemas com clocks dessincronizados
            ClockSkew = TimeSpan.Zero
        };
    }

    /// <summary>
    /// Valida uma string de token JWT.
    /// </summary>
    /// <param name="token">A string do token (sem o prefixo "Bearer ").</param>
    /// <returns>O ClaimsPrincipal se o token for válido; caso contrário, retorna null.</returns>
    public ClaimsPrincipal? ValidateToken(string token)
    {
        if (string.IsNullOrEmpty(token))
        {
            return null;
        }

        var tokenHandler = new JwtSecurityTokenHandler();
        try
        {
            // Tenta validar o token usando os parâmetros configurados
            var principal = tokenHandler.ValidateToken(token, _tokenValidationParameters, out SecurityToken validatedToken);
            
            _logger.LogInformation("Token JWT validado com sucesso para o usuário/nó: {Name}", principal.Identity?.Name);
            return principal;
        }
        catch (SecurityTokenException ex)
        {
            // A validação falhou por um motivo específico (expirado, assinatura inválida, etc.)
            _logger.LogWarning("Falha na validação do token JWT: {Message}", ex.Message);
            return null;
        }
        catch (Exception ex)
        {
            // Outro erro inesperado durante a validação
            _logger.LogError(ex, "Ocorreu um erro inesperado durante a validação do token.");
            return null;
        }
    }
}