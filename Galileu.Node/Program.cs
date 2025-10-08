using System.Net;
using Galileu.Node.Models;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Hosting;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using System.Net.Http.Json;
using System.Net.Sockets;
using Galileu.Node.Gpu;
using Galileu.Node.Services;
using Microsoft.AspNetCore.Http;
using Services;
using System.Globalization;
using Galileu.Node.Core;

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
Console.WriteLine($" === Swagger Acess http://localhost:{port}/swagger/index.html ===");
builder.WebHost.UseUrls(myAddress);
builder.Services.AddSingleton<OpenCLService>();
builder.Services.AddSingleton<PrimingService>();
builder.Services.AddSingleton(new NodeState(myAddress));
builder.Services.AddSingleton<PolymorphicTypeResolver>();
builder.Services.AddSingleton(provider => new NodeClient(provider.GetRequiredService<PolymorphicTypeResolver>()));
builder.Services.AddHostedService<GossipService>();
builder.Services.AddSingleton<NodeRegistryService>();
builder.Services.AddSingleton<GenerativeService>();
builder.Services.AddSingleton<WalletService>();
builder.Services.AddSingleton<ActorSystemSingleton>();
builder.Services.AddHostedService<AkkaHostedService>();
builder.Services.AddControllers();

builder.Services.AddEndpointsApiExplorer();

builder.Services.AddSwaggerGen(options =>
{
    options.SwaggerDoc("v1", new Microsoft.OpenApi.Models.OpenApiInfo
    {
        Title = "Dyson Node API",
        Version = "v1"
    });
});

var app = builder.Build();

app.UseSwagger();
app.UseSwaggerUI(options =>
{
    options.SwaggerEndpoint("/swagger/v1/swagger.json", "Dyson Node API V1");
    options.RoutePrefix = "swagger";
});

app.UseWebSockets();
app.MapControllers();

app.MapGet("/ws", async (HttpContext context) => { });

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

_ = BootstrapNodeAsync(app.Services, args);

app.Run();


async Task BootstrapNodeAsync(IServiceProvider services, string[] args)
{
    var nodeState = services.GetRequiredService<NodeState>();
    await Task.Delay(2000);

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
    
    var _generativeService = services.GetRequiredService<GenerativeService>();
    string modelPath = Path.Combine(Environment.CurrentDirectory, "Dayson", "Dayson.json");
    var datasetPath = Path.Combine(Environment.CurrentDirectory, "Dayson", "pt_0.txt");
    
    if (!File.Exists(modelPath))
    {
        if (!File.Exists(datasetPath))
        {
            Console.WriteLine($"Arquivo de dataset não encontrado em: {datasetPath}");
            return;
        }

        Console.WriteLine("\n========================================");
        Console.WriteLine("CONFIGURAÇÃO DE TREINAMENTO OTIMIZADA");
        Console.WriteLine("========================================");

        var vocabManager = new VocabularyManager();
        
        // ===== VOCABULÁRIO OTIMIZADO =====
        // Reduzido para 10.000 tokens (em vez de 50.000+)
        // Isso reduz o tamanho da camada de saída em 80%
        int vocabSize = vocabManager.BuildVocabulary(datasetPath, maxVocabSize: 20000);
        if (vocabSize == 0)
        {
            Console.WriteLine("Falha ao construir ou carregar o vocabulário.");
            return;
        }
        var trainer = new Trainer(
            datasetPath, 
            epochs: 50,           // Reduzido de 100 para 30
            learningRate: 0.0001,  // Aumentado de 0.0001 para 0.001 (convergência mais rápida)
            validationSplit: 0.2,
            batchSize: 32
        );
        Console.WriteLine($"\n[Config] Vocabulário: {vocabSize} tokens");
        Console.WriteLine($"[Config] Hidden Size: 256");
        Console.WriteLine($"[Config] Context Window: 3 timesteps");
        Console.WriteLine($"[Config] Épocas: {trainer.epochs}");
        Console.WriteLine($"[Config] Learning Rate: {trainer.learningRate}");
        Console.WriteLine($"[Config] Batch Size: {trainer.batchSize}");
        Console.WriteLine($"[Config] Validação: {trainer.validationSplit *100}%\n");

        Console.WriteLine("========================================");
        Console.WriteLine("INICIANDO TREINAMENTO");
        Console.WriteLine("Tempo estimado: 8-16 horas (CPU)");
        Console.WriteLine("========================================\n");

        await _generativeService.TrainModelAsync(trainer);
        Console.WriteLine("\n========================================");
        Console.WriteLine("TREINAMENTO CONCLUÍDO!");
        Console.WriteLine("========================================");
    }
    else
    {
        Console.WriteLine($"[Bootstrap] Modelo encontrado em {modelPath}. Pulando treinamento.");
        _generativeService.InitializeFromDisk();
    }
}