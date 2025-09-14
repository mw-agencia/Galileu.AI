using Galileu.Models;
using Galileu.Services;
using Microsoft.AspNetCore.Authentication.JwtBearer;
using Microsoft.IdentityModel.Tokens;
using System.Text;
using System.Text.Json;
using System.Net.WebSockets;

var builder = WebApplication.CreateBuilder(args);

// --- 1. Configuração dos Serviços ---
var port = args.Length > 0 && args[0] == "root" ? args.ElementAtOrDefault(1) ?? "5001" : args.ElementAtOrDefault(1) ?? "5002";
var myAddress = $"http://localhost:{port}";

builder.WebHost.UseUrls(myAddress);

// --- Registro dos Serviços (Forma correta e sem duplicatas) ---
builder.Services.Configure<MongoDbSettings>(builder.Configuration.GetSection("MongoDbSettings"));
builder.Services.AddSingleton(new NodeState(myAddress)); // Usa o NodeState da versão Gossip
builder.Services.AddSingleton<NodeClient>();
builder.Services.AddSingleton<PolymorphicTypeResolver>();
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

app.MapGet("/ws", async (HttpContext context, PolymorphicTypeResolver typeResolver, NodeState state, NodeClient client) =>
{
    // ... (A lógica do loop do WebSocket permanece a mesma) ...
});

// --- Lógica do HandleMessage ATUALIZADA para Gossip ---
async Task<Message?> HandleMessage(Message? receivedMessage, NodeState state, NodeClient client)
{
    switch (receivedMessage)
    {
        // Este continua relevante para monitoramento
        case PingRequest ping:
            Console.WriteLine($"[Server] Received ping from {ping.FromNodeId}.");
            return new PongResponse(ping.CorrelationId, $"Pong from {state.Id}");

        // NOVO: Handler para o protocolo Gossip
        case GossipSyncRequest gossipRequest:
            state.MergePeers(gossipRequest.KnownPeers);
            var ourPeers = state.GetKnownPeers();
            return new GossipSyncResponse(gossipRequest.CorrelationId, ourPeers);
            
        // REMOVIDO: Os handlers para JoinRequest e ForwardJoinRequest foram removidos por serem obsoletos.
            
        default:
            Console.WriteLine($"[Server] Received message of unhandled type: {receivedMessage?.GetType().Name}.");
            return null;
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