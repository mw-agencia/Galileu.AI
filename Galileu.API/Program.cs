using System.Net.WebSockets;
using System.Text;
using System.Text.Json;
using Galileu.Models;
using Galileu.Services;
using Microsoft.AspNetCore.Authentication.JwtBearer;
using Microsoft.IdentityModel.Tokens; // Necessário para IHostBuilder

var builder = WebApplication.CreateBuilder(args);

// --- 1. Configuração dos Serviços (Tudo junto) ---
var port = args.Length > 0 && args[0] == "root" ? args.ElementAtOrDefault(1) ?? "5001" : args.ElementAtOrDefault(1) ?? "5002";
var myAddress = $"http://localhost:{port}";

// Configura o Kestrel (servidor web) para ouvir na URL e porta corretas
builder.WebHost.UseUrls(myAddress);

// Instanciando os objetos que serão registrados como singletons
var nodeState = new NodeState(myAddress);
var nodeClient = new NodeClient();

// Registro dos serviços da aplicação (APENAS UMA VEZ)
builder.Services.Configure<MongoDbSettings>(builder.Configuration.GetSection("MongoDbSettings"));
builder.Services.AddSingleton(new NodeState(myAddress));
builder.Services.AddSingleton<NodeClient>();
builder.Services.AddSingleton<PolymorphicTypeResolver>();
builder.Services.AddSingleton<WalletService>();
builder.Services.AddSingleton<RewardContractService>();
builder.Services.AddSingleton<NodeRegistryService>();

// --- Lógica do Akka.NET ---
// Registra o Hosted Service que vai gerenciar o Akka
builder.Services.AddHostedService<AkkaHostedService>();
// Registra um singleton para que possamos acessar o ActorSystem de outros serviços
builder.Services.AddSingleton<ActorSystemSingleton>();


// Serviços para API e Swagger
builder.Services.AddControllers();
builder.Services.AddEndpointsApiExplorer();
builder.Services.AddAuthentication(JwtBearerDefaults.AuthenticationScheme)
    .AddJwtBearer(options =>
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

// --- 3. Configuração do Pipeline de Middlewares (A ordem importa!) ---
// Habilitar o Swagger apenas em ambiente de desenvolvimento é uma boa prática
if (app.Environment.IsDevelopment())
{
    app.UseSwagger();
    app.UseSwaggerUI(options =>
    {
        options.SwaggerEndpoint("/swagger/v1/swagger.json", "Galileu.AI Nodes API");
        // Rota para acessar a UI do Swagger (ex: http://localhost:5001/swagger)
        options.RoutePrefix = "swagger";
    });
}

app.UseWebSockets();

// Mapeia os endpoints ANTES de rodar a aplicação
app.MapControllers(); 

app.MapGet("/", () => $"Galileu.AI is running at {myAddress}.");

app.MapGet("/ws", async (HttpContext context, PolymorphicTypeResolver typeResolver, NodeState state, NodeClient client) =>
{
    if (context.WebSockets.IsWebSocketRequest)
    {
        using var ws = await context.WebSockets.AcceptWebSocketAsync();
        Console.WriteLine($"[Server] WebSocket connection established from {context.Connection.RemoteIpAddress}.");

        var buffer = new byte[1024 * 4];
        while (ws.State == WebSocketState.Open)
        {
            var result = await ws.ReceiveAsync(new ArraySegment<byte>(buffer), CancellationToken.None);
            if (result.MessageType == WebSocketMessageType.Close)
            {
                await ws.CloseAsync(result.CloseStatus.Value, result.CloseStatusDescription, CancellationToken.None);
                break;
            }

            var jsonMessage = Encoding.UTF8.GetString(buffer, 0, result.Count);
            
            var options = new JsonSerializerOptions { TypeInfoResolver = typeResolver };
            var message = JsonSerializer.Deserialize<Message>(jsonMessage, options);

            // Passa as dependências para o handler
            Message? response = await HandleMessage(message, state, client);
            
            if (response != null)
            {
                var jsonResponse = JsonSerializer.Serialize(response, response.GetType(), options);
                var responseBuffer = Encoding.UTF8.GetBytes(jsonResponse);
                await ws.SendAsync(new ArraySegment<byte>(responseBuffer), WebSocketMessageType.Text, true, CancellationToken.None);
            }
        }
    }
    else
    {
        context.Response.StatusCode = 400;
    }
});

// A função HandleMessage pode ser uma função local dentro do Program.cs
async Task<Message?> HandleMessage(Message? receivedMessage, NodeState state, NodeClient client)
{
    switch (receivedMessage)
    {
        case PingRequest ping:
            Console.WriteLine($"[Server] Received ping from {ping.FromNodeId}.");
            return new PongResponse(ping.CorrelationId, $"Pong from {state.Id}");

        case JoinRequest join:
            Console.WriteLine($"[Server] Node {join.NewNodeId} at {join.NewNodeAddress} requested to join.");
            if (state.LeftChildAddress == null)
            {
                state.LeftChildAddress = join.NewNodeAddress;
                state.PrintStatus();
                return new JoinResponse(join.CorrelationId, true, state.Id, "Joined as left child.");
            }
            if (state.RightChildAddress == null)
            {
                state.RightChildAddress = join.NewNodeAddress;
                state.PrintStatus();
                return new JoinResponse(join.CorrelationId, true, state.Id, "Joined as right child.");
            }
            
            Console.WriteLine("[Server] Both children occupied. Forwarding request...");
            var forwardRequest = new ForwardJoinRequest(Guid.NewGuid(), join);
            var forwardResponse = await client.SendRequestAsync<JoinResponse>(state.LeftChildAddress, forwardRequest, CancellationToken.None);
            return new JoinResponse(join.CorrelationId, forwardResponse.Success, forwardResponse.ParentNodeId, forwardResponse.Message);
            
        case ForwardJoinRequest forward:
            Console.WriteLine($"[Server] Received forwarded join request for {forward.OriginalRequest.NewNodeId}.");
            return await HandleMessage(forward.OriginalRequest, state, client);
            
        default:
            Console.WriteLine($"[Server] Unknown message type received.");
            return null;
    }
}


// --- 4. Lógica de Inicialização (Join na rede) ---
var isRoot = args.Length > 0 && args[0].Equals("root", StringComparison.OrdinalIgnoreCase);

Console.WriteLine($"Node {nodeState.Id} is running at {myAddress}");
if (isRoot)
{
    Console.WriteLine("This node is the ROOT of the tree.");
}
else
{
    // A lógica de Join roda em background para não bloquear o início do servidor
    _ = Task.Run(async () =>
    {
        // Espera um pouco para garantir que o nó raiz esteja ouvindo
        await Task.Delay(2000); 
        var bootstrapAddress = args.ElementAtOrDefault(0) ?? "http://localhost:5001";
        Console.WriteLine($"Attempting to join tree via bootstrap node: {bootstrapAddress}...");
        try
        {
            var joinRequest = new JoinRequest(Guid.NewGuid(), nodeState.Id, nodeState.Address);
            var response = await nodeClient.SendRequestAsync<JoinResponse>(bootstrapAddress, joinRequest, CancellationToken.None);
            
            if (response.Success)
            {
                nodeState.ParentAddress = bootstrapAddress; // Simplificação
                Console.WriteLine($"Successfully joined the tree! Parent: {response.ParentNodeId}");
            }
            else
            {
                Console.WriteLine($"Failed to join the tree: {response.Message}");
            }
            nodeState.PrintStatus();
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Error joining tree: {ex.Message}");
        }
    });
}
nodeState.PrintStatus();

// --- 5. Executar a Aplicação (Esta é a última chamada e ela é bloqueante) ---
app.Run();