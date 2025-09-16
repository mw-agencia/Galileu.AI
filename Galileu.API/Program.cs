using Galileu.Models;
using Galileu.Services;
using Microsoft.AspNetCore.Authentication.JwtBearer;
using Microsoft.IdentityModel.Tokens;
using System.Text;
using System.Text.Json;
using System.Net.WebSockets;
using System.Security.Claims;

var builder = WebApplication.CreateBuilder(args);

// --- 1. Configuração dos Serviços ---
var port = args.Length > 0 && args[0] == "root" ? args.ElementAtOrDefault(1) ?? "5001" : args.ElementAtOrDefault(1) ?? "5002";
var myAddress = $"http://localhost:{port}";

builder.WebHost.UseUrls(myAddress);

var typeResolver = new PolymorphicTypeResolver();

// --- Registro dos Serviços (Forma correta e sem duplicatas) ---
builder.Services.Configure<MongoDbSettings>(builder.Configuration.GetSection("MongoDbSettings"));
builder.Services.AddSingleton(new NodeState(myAddress)); // Usa o NodeState da versão Gossip
builder.Services.AddSingleton<NodeClient>();
builder.Services.AddSingleton<PolymorphicTypeResolver>();
builder.Services.AddSingleton(typeResolver);
builder.Services.AddSingleton<WalletService>();
builder.Services.AddSingleton<RewardContractService>();
builder.Services.AddSingleton<NodeRegistryService>();

// --- Lógica do Akka.NET ---
builder.Services.AddHostedService<AkkaHostedService>();
builder.Services.AddSingleton<ActorSystemSingleton>();

// --- NOVO: Adiciona o GossipService e remove o HealthCheckService antigo ---
builder.Services.AddHostedService<GossipService>();
// A linha builder.Services.AddHostedService<HealthCheckService>(); foi removida.
builder.Services.AddAuthentication(options =>
    {
        options.DefaultAuthenticateScheme = JwtBearerDefaults.AuthenticationScheme;
        options.DefaultChallengeScheme = JwtBearerDefaults.AuthenticationScheme;
    })
    .AddJwtBearer(options => // O .AddJwtBearer() continua o mesmo
    {
        options.TokenValidationParameters = new TokenValidationParameters
        {
            ValidateIssuer = true,
            ValidateAudience = true,
            ValidateLifetime = true,
            ValidateIssuerSigningKey = true,
            ValidIssuer = "GalileuAPI",
            ValidAudience = "GalileuUsers",
            IssuerSigningKey = new SymmetricSecurityKey(Encoding.UTF8.GetBytes("SUA_CHAVE_SECRETA_SUPER_LONGA_E_SEGURA_AQUI"))
        };
    });

builder.Services.AddAuthorization();
// Serviços para API e Swagger
builder.Services.AddControllers();
builder.Services.AddEndpointsApiExplorer();
// ... (Sua configuração de Autenticação e Swagger continua a mesma) ...
builder.Services.AddSwaggerGen(options =>
{
    options.SwaggerDoc("v1", new Microsoft.OpenApi.Models.OpenApiInfo
    {
        Title = "Galileu P2P Node API",
        Version = "v1",
        Description = "API para interagir e monitorar um nó na rede distribuída Galileu."
    });
});


// --- 2. Construção da Aplicação ---
var app = builder.Build();

// --- 3. Configuração do Pipeline de Middlewares ---
if (app.Environment.IsDevelopment())
{
    app.UseSwagger();
    app.UseSwaggerUI(options =>
    {
        options.SwaggerEndpoint("/swagger/v1/swagger.json", "Galileu Node API V1");
        options.RoutePrefix = "swagger";
    });
}
app.UseAuthentication();
app.UseAuthorization();
app.UseWebSockets();
app.MapControllers(); 

app.MapGet("/", () => $"Galileu P2P Node (Gossip Protocol) is running at {myAddress}.");

app.MapGet("/ws", async (HttpContext context) =>
{
    if (!context.WebSockets.IsWebSocketRequest)
    {
        context.Response.StatusCode = 400;
        return;
    }
    
    using var ws = await context.WebSockets.AcceptWebSocketAsync();
    Console.WriteLine($"[Server] Nova conexão de {context.Connection.RemoteIpAddress}. Aguardando handshake...");

    // 1. Tenta autenticar a conexão via handshake
    var (isAuthenticated, principal) = await AuthenticateHandshake(ws, context.RequestServices);
    if (!isAuthenticated)
    {
        Console.WriteLine($"[Server] Handshake falhou para {context.Connection.RemoteIpAddress}. Conexão encerrada.");
        return; // A conexão já foi fechada dentro do AuthenticateHandshake
    }

    Console.WriteLine($"[Server] Handshake bem-sucedido para o nó '{principal?.Identity?.Name}'. Iniciando loop de mensagens.");
    
    // 2. Se o handshake for bem-sucedido, inicia o loop de mensagens normal
    try
    {
        var buffer = new byte[4096];
        while (ws.State == WebSocketState.Open)
        {
            var result = await ws.ReceiveAsync(new ArraySegment<byte>(buffer), CancellationToken.None);
            if (result.MessageType == WebSocketMessageType.Close) break;

            var jsonMessage = Encoding.UTF8.GetString(buffer, 0, result.Count);
            
            var typeResolver = context.RequestServices.GetRequiredService<PolymorphicTypeResolver>();
            var options = new JsonSerializerOptions { TypeInfoResolver = typeResolver };
            var message = JsonSerializer.Deserialize<Message>(jsonMessage, options);
            
            var state = context.RequestServices.GetRequiredService<NodeState>();
            var client = context.RequestServices.GetRequiredService<NodeClient>();
            Message? response = await HandleMessage(message, state, client);
            
            if (response != null)
            {
                var jsonResponse = JsonSerializer.Serialize(response, response.GetType(), options);
                var responseBuffer = Encoding.UTF8.GetBytes(jsonResponse);
                await ws.SendAsync(new ArraySegment<byte>(responseBuffer), WebSocketMessageType.Text, true, CancellationToken.None);
            }
        }
    }
    catch (WebSocketException ex) when (ex.WebSocketErrorCode == WebSocketError.ConnectionClosedPrematurely)
    {
        Console.WriteLine($"[Server] Conexão com o nó '{principal?.Identity?.Name}' fechada abruptamente.");
    }
    catch (Exception ex)
    {
        Console.WriteLine($"[Server] Erro inesperado na conexão com '{principal?.Identity?.Name}': {ex.Message}");
    }
    finally
    {
        Console.WriteLine($"[Server] Conexão com '{principal?.Identity?.Name}' finalizada.");
        ws.Dispose();
    }
});

// --- Lógica do HandleMessage ATUALIZADA para Gossip ---
// A função HandleMessage pode permanecer a mesma, mas simplificada
Task<Message?> HandleMessage(Message? receivedMessage, NodeState state, NodeClient client)
{
    switch (receivedMessage)
    {
        case PingRequest ping:
            Console.WriteLine($"[Server] Received ping from {ping.FromNodeId}.");
            return Task.FromResult<Message?>(new PongResponse(ping.CorrelationId, $"Pong from {state.Id}"));

        case GossipSyncRequest gossipRequest:
            state.MergePeers(gossipRequest.KnownPeers);
            var ourPeers = state.GetKnownPeers();
            return Task.FromResult<Message?>(new GossipSyncResponse(gossipRequest.CorrelationId, ourPeers));
            
        default:
            Console.WriteLine($"[Server] Received message of unhandled type: {receivedMessage?.GetType().Name}.");
            return Task.FromResult<Message?>(null);
    }
}

async Task<(bool, ClaimsPrincipal?)> AuthenticateHandshake(WebSocket ws, IServiceProvider services)
{
    var typeResolver = services.GetRequiredService<PolymorphicTypeResolver>();
    var jwtValidator = services.GetRequiredService<JwtValidatorService>();
    var options = new JsonSerializerOptions { TypeInfoResolver = typeResolver };
    
    var buffer = new byte[2048]; // Buffer para a mensagem de auth
    try
    {
        // Espera a primeira mensagem (AuthRequest) por até 10 segundos
        using var cts = new CancellationTokenSource(TimeSpan.FromSeconds(10));
        var result = await ws.ReceiveAsync(new ArraySegment<byte>(buffer), cts.Token);
        
        var jsonMessage = Encoding.UTF8.GetString(buffer, 0, result.Count);
        if (JsonSerializer.Deserialize<Message>(jsonMessage, options) is not AuthRequest authRequest)
        {
            await ws.CloseAsync(WebSocketCloseStatus.PolicyViolation, "Primeira mensagem deve ser um AuthRequest.", CancellationToken.None);
            return (false, null);
        }

        var principal = jwtValidator.ValidateToken(authRequest.NodeJwt);
        if (principal == null || !principal.IsInRole("node"))
        {
            var failResponse = new AuthResponse(authRequest.CorrelationId, false, "Token JWT inválido ou sem a role 'node'.");
            var jsonFailResponse = JsonSerializer.Serialize(failResponse, options);
            await ws.SendAsync(new ArraySegment<byte>(Encoding.UTF8.GetBytes(jsonFailResponse)), WebSocketMessageType.Text, true, CancellationToken.None);
            await ws.CloseAsync(WebSocketCloseStatus.PolicyViolation, "Token inválido.", CancellationToken.None);
            return (false, null);
        }

        // Sucesso!
        var successResponse = new AuthResponse(authRequest.CorrelationId, true, "Autenticação bem-sucedida.");
        var jsonSuccessResponse = JsonSerializer.Serialize(successResponse, options);
        await ws.SendAsync(new ArraySegment<byte>(Encoding.UTF8.GetBytes(jsonSuccessResponse)), WebSocketMessageType.Text, true, CancellationToken.None);
        
        return (true, principal);
    }
    catch (Exception ex) // Captura timeouts, JSON inválido, etc.
    {
        Console.WriteLine($"Erro durante o handshake: {ex.Message}");
        if (ws.State == WebSocketState.Open)
        {
            await ws.CloseAsync(WebSocketCloseStatus.InternalServerError, "Erro no handshake.", CancellationToken.None);
        }
        return (false, null);
    }
}

async Task BootstrapNodeAsync(IServiceProvider services, string[] args)
{
    var nodeState = services.GetRequiredService<NodeState>();
    
    // Espera um pouco para o host web iniciar completamente
    await Task.Delay(2000);

    var bootstrapAddress = args.ElementAtOrDefault(0) ?? "http://localhost:5001";
    Console.WriteLine($"Iniciando processo de registro e bootstrap com o Orquestrador: {bootstrapAddress}...");

    // 1. Gera sua própria carteira
    var (publicKey, _) = CryptoUtils.GenerateKeyPair();
    var normalizedPublicKey = CryptoUtils.NormalizePublicKey(publicKey);

    // 2. Registra-se no Orquestrador para obter JWT e lista de pares
    using var apiClient = new HttpClient { BaseAddress = new Uri(bootstrapAddress) };
    try
    {
        var registrationRequest = new NodeRegistrationRequest(normalizedPublicKey, nodeState.Address);
        var response = await apiClient.PostAsJsonAsync("/api/auth/register-node", registrationRequest);
        
        if (!response.IsSuccessStatusCode)
        {
            Console.WriteLine($"ERRO: Falha ao registrar no orquestrador. Status: {response.StatusCode}. O nó permanecerá offline.");
            // Em um sistema real, aqui entraria uma lógica de retry.
            return;
        }

        var registrationResponse = await response.Content.ReadFromJsonAsync<NodeRegistrationResponse>();
        if (string.IsNullOrEmpty(registrationResponse?.NodeJwt))
        {
            Console.WriteLine("ERRO: O orquestrador não retornou um JWT válido. O nó permanecerá offline.");
            return;
        }

        // 3. Armazena as credenciais e a lista de pares inicial
        nodeState.NodeJwt = registrationResponse.NodeJwt;
        nodeState.MergePeers(registrationResponse.InitialPeers);
        
        Console.WriteLine("Nó registrado com sucesso. JWT recebido e lista de pares inicial populada.");
        nodeState.PrintStatus();
        
        // O GossipService, que já está rodando, agora tem pares para começar a fofocar.
    }
    catch (Exception ex)
    {
        Console.WriteLine($"ERRO CRÍTICO durante o bootstrap: {ex.Message}. O nó tentará novamente se a aplicação for reiniciada.");
    }
}


// --- 4. Lógica de Inicialização ATUALIZADA para Gossip ---
var nodeState = app.Services.GetRequiredService<NodeState>();
var isRoot = args.Length > 0 && args[0].Equals("root", StringComparison.OrdinalIgnoreCase);

Console.WriteLine($"Node {nodeState.Id} is running at {myAddress}");
nodeState.PrintStatus();

if (!isRoot)
{
    // A lógica de "join" agora é apenas saber de um nó de bootstrap
    var bootstrapAddress = args.ElementAtOrDefault(0) ?? "http://localhost:5001";
    Console.WriteLine($"Bootstrapping into the network via: {bootstrapAddress}...");
    nodeState.MergePeers(new[] { bootstrapAddress });
}


// --- 5. Executar a Aplicação ---
app.Run();