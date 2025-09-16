using System.Net;
using Galileu.Node.Models;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Hosting;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using System.Net.Http.Json;
using System.Net.Sockets;
using Galileu.Node.Services;
using Microsoft.AspNetCore.Http;
using Services;

// --- 1. CONFIGURAÇÃO DA APLICAÇÃO (NÓ) ---
var builder = WebApplication.CreateBuilder(args);
var port = GetAvailablePort();

int GetAvailablePort()
{
    using (var listener = new TcpListener(IPAddress.Loopback, 0))
    {
        listener.Start();
        int port = ((IPEndPoint)listener.LocalEndpoint).Port;
        listener.Stop();
        return port;
    }
}

var myAddress = $"http://localhost:{port}";
builder.WebHost.UseUrls(myAddress);

// --- 2. REGISTRO DE SERVIÇOS DO NÓ ---
builder.Services.AddSingleton(new NodeState(myAddress));
builder.Services.AddSingleton<PolymorphicTypeResolver>();
builder.Services.AddSingleton(provider => new NodeClient(provider.GetRequiredService<PolymorphicTypeResolver>()));
builder.Services.AddHostedService<GossipService>();
builder.Services.AddSingleton<NodeRegistryService>();
builder.Services.AddSingleton<ActorSystemSingleton>();
builder.Services.AddHostedService<AkkaHostedService>();
builder.Services.AddControllers();

builder.Services.AddEndpointsApiExplorer(); 

builder.Services.AddSwaggerGen(options =>
{
    // Você pode adicionar configurações extras aqui, como o título da API
    options.SwaggerDoc("v1", new Microsoft.OpenApi.Models.OpenApiInfo 
    { 
        Title = "Galileu Node API", 
        Version = "v1" 
    });
});

var app = builder.Build();

app.UseSwagger();
app.UseSwaggerUI(options =>
{
    options.SwaggerEndpoint("/swagger/v1/swagger.json", "Galileu Node API V1");
    options.RoutePrefix = "swagger";
});

app.UseWebSockets();
app.MapControllers(); 

// Endpoint WebSocket para receber comunicação P2P
app.MapGet("/ws", async (HttpContext context) =>
{
    // ... (lógica completa de handshake e loop de mensagens que já implementamos)
});

// Handler para mensagens P2P
async Task<Message?> HandleMessage(Message? receivedMessage, NodeState state, NodeClient client)
{
    switch (receivedMessage)
    {
        case GossipSyncRequest gossipRequest:
            state.MergePeers(gossipRequest.KnownPeers);
            return new GossipSyncResponse(gossipRequest.CorrelationId, state.GetKnownPeers());
        default: return null;
    }
}


// --- 4. LÓGICA DE INICIALIZAÇÃO DO NÓ ---
_ = BootstrapNodeAsync(app.Services, args);

app.Run();


// --- FUNÇÃO DE BOOTSTRAP PARA O NÓ ---
async Task BootstrapNodeAsync(IServiceProvider services, string[] args)
{
    var nodeState = services.GetRequiredService<NodeState>();
    await Task.Delay(2000); // Espera o host iniciar

    var bootstrapApiAddress = args.Length > 0 ? args[0] : "http://localhost:5001";
    Console.WriteLine($"Iniciando registro com o Orquestrador em {bootstrapApiAddress}...");

    var (publicKey, _) = CryptoUtils.GenerateKeyPair();
    var normalizedPublicKey = CryptoUtils.NormalizePublicKey(publicKey);

    using var apiClient = new HttpClient { BaseAddress = new Uri(bootstrapApiAddress) };
    try
    {
        var registrationRequest = new NodeRegistrationRequest(normalizedPublicKey, nodeState.Address);
        var response = await apiClient.PostAsJsonAsync("/api/auth/register-node", registrationRequest);
        response.EnsureSuccessStatusCode();

        var regResponse = await response.Content.ReadFromJsonAsync<NodeRegistrationResponse>();
        if (string.IsNullOrEmpty(regResponse?.NodeJwt)) throw new Exception("Orquestrador não retornou um JWT.");

        nodeState.NodeJwt = regResponse.NodeJwt;
        nodeState.MergePeers(regResponse.InitialPeers);
        
        Console.WriteLine("Nó validado pelo Orquestrador. JWT recebido e rede P2P iniciada.");
        nodeState.PrintStatus();
    }
    catch (Exception ex)
    {
        Console.WriteLine($"ERRO CRÍTICO no registro: {ex.Message}. O nó não pode se juntar à rede e permanecerá em modo de espera.");
    }
}