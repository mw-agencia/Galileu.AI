using Akka.IO;
using Galileu.Node.Models;
using Galileu.Services;
using Services;
using Microsoft.AspNetCore.Authentication.JwtBearer;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Hosting;
using Microsoft.AspNetCore.Http;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Microsoft.IdentityModel.Tokens;
using Services;

// --- 1. CONFIGURAÇÃO DA APLICAÇÃO (NÓ) ---
var builder = WebApplication.CreateBuilder(args);

// Lê os argumentos da linha de comando para definir a identidade do nó
var port = args.Length > 0 && args[0] == "root" ? args.ElementAtOrDefault(1) ?? "5001" : args.ElementAtOrDefault(1) ?? "5002";
var myAddress = $"http://localhost:{port}";

// Configura o servidor web para ouvir no endereço correto
builder.WebHost.UseUrls(myAddress);

// --- 2. REGISTRO DE SERVIÇOS DO NÓ ---

// Configuração do MongoDB

// Serviços P2P e de Estado
builder.Services.AddSingleton(new NodeState(myAddress));
builder.Services.AddSingleton<PolymorphicTypeResolver>();
builder.Services.AddSingleton(provider => new NodeClient(provider.GetRequiredService<PolymorphicTypeResolver>()));
builder.Services.AddHostedService<GossipService>(); // Para descoberta de pares

// Serviços de Token e Recompensa
//builder.Services.AddSingleton<WalletService>();
//builder.Services.AddSingleton<RewardContractService>();

// Serviços do Sistema de Atores (Akka.NET)
//builder.Services.AddSingleton<NodeRegistryService>();
//builder.Services.AddSingleton<ActorSystemSingleton>();
builder.Services.AddHostedService<AkkaHostedService>(); // Gerencia o ciclo de vida do Akka.NET e cria os atores

// Serviços para API HTTP (Swagger, Controllers)
builder.Services.AddControllers();
builder.Services.AddEndpointsApiExplorer();
builder.Services.AddSwaggerGen(options => { /* ... */ });
builder.Services.AddAuthentication(options => { /* ... */ }).AddJwtBearer(options => { /* ... */ });
builder.Services.AddAuthorization();


// --- 3. CONSTRUÇÃO E CONFIGURAÇÃO DO PIPELINE ---
var app = builder.Build();

// Middlewares
if (app.Environment.IsDevelopment())
{
    app.UseSwagger();
    app.UseSwaggerUI(options => { /* ... */ });
}
app.UseAuthentication();
app.UseAuthorization();
app.UseWebSockets(); // Habilita o protocolo WebSocket

// Endpoints
app.MapControllers(); 
app.MapGet("/", () => $"Nó Galileu.AI (P2P Gossip) rodando em {myAddress}.");

// Endpoint WebSocket: a "porta de entrada" para comunicação de baixo nível da rede
// Handler para mensagens da rede P2P
async Task<Message?> HandleMessage(Message? receivedMessage, NodeState state, NodeClient client)
{
    switch (receivedMessage)
    {
        case GossipSyncRequest gossipRequest:
            state.MergePeers(gossipRequest.KnownPeers);
            return new GossipSyncResponse(gossipRequest.CorrelationId, state.GetKnownPeers());
            
        default:
            Console.WriteLine($"[Server] Mensagem P2P de tipo não tratado recebida: {receivedMessage?.GetType().Name}.");
            return null;
    }
}

// --- 4. LÓGICA DE INICIALIZAÇÃO DO NÓ ---
var nodeState = app.Services.GetRequiredService<NodeState>();
var isRoot = args.Length > 0 && args[0].Equals("root", StringComparison.OrdinalIgnoreCase);

Console.WriteLine($"Nó {nodeState.Id} rodando em {myAddress}");
if (isRoot)
{
    Console.WriteLine("Este nó é o primeiro na rede (bootstrap).");
}
else
{
    var bootstrapAddress = args.ElementAtOrDefault(0) ?? "http://localhost:5001";
    Console.WriteLine($"Tentando entrar na rede via bootstrap: {bootstrapAddress}...");
    nodeState.MergePeers(new[] { bootstrapAddress });
}
nodeState.PrintStatus();

// --- 5. EXECUÇÃO DO NÓ ---
app.Run();