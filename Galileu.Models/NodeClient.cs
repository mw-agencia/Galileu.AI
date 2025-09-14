// Galileu.Models/NodeClient.cs
using System.Collections.Concurrent;
using System.Net.WebSockets;
using System.Text;
using System.Text.Json;

namespace Galileu.Models;

public class NodeClient : IDisposable
{
    // O dicionário agora armazena uma Task<ClientWebSocket> para gerenciar a conexão.
    private readonly ConcurrentDictionary<string, ClientWebSocket> _sockets = new();
    private readonly ConcurrentDictionary<Guid, TaskCompletionSource<Message>> _pendingRequests = new();
    private readonly PolymorphicTypeResolver _typeResolver;
    private readonly JsonSerializerOptions _jsonOptions;

    public NodeClient(PolymorphicTypeResolver typeResolver)
    {
        _typeResolver = typeResolver;
        _jsonOptions = new JsonSerializerOptions { TypeInfoResolver = _typeResolver };
    }

    public async Task<TResponse> SendRequestAsync<TResponse>(string targetAddress, Message request, CancellationToken token) where TResponse : Message
    {
        // Usa um ClientWebSocket por requisição para máxima robustez.
        using var ws = new ClientWebSocket();
        var wsUrl = new Uri(targetAddress.Replace("http://", "ws://").Replace("https://", "wss://") + "/ws");

        // Conecta ao servidor
        await ws.ConnectAsync(wsUrl, token);

        // Envia a mensagem
        var jsonMessage = JsonSerializer.Serialize(request, _jsonOptions);
        var buffer = Encoding.UTF8.GetBytes(jsonMessage);
        await ws.SendAsync(new ArraySegment<byte>(buffer), WebSocketMessageType.Text, true, token);

        // Aguarda a resposta
        var receiveBuffer = new byte[1024 * 4];
        var result = await ws.ReceiveAsync(new ArraySegment<byte>(receiveBuffer), token);

        var jsonResponse = Encoding.UTF8.GetString(receiveBuffer, 0, result.Count);
        var responseMessage = JsonSerializer.Deserialize<Message>(jsonResponse, _jsonOptions);

        return (TResponse)responseMessage!;
    }

    private async Task ReceiveLoop(ClientWebSocket ws, string url)
    {
        var buffer = new byte[1024 * 4];
        try
        {
            while (ws.State == WebSocketState.Open)
            {
                var result = await ws.ReceiveAsync(new ArraySegment<byte>(buffer), CancellationToken.None);
                if (result.MessageType == WebSocketMessageType.Close) break;

                var jsonResponse = Encoding.UTF8.GetString(buffer, 0, result.Count);
                var responseMessage = JsonSerializer.Deserialize<Message>(jsonResponse, new JsonSerializerOptions { TypeInfoResolver = new PolymorphicTypeResolver() });

                if (responseMessage != null && _pendingRequests.TryRemove(responseMessage.CorrelationId, out var tcs))
                {
                    tcs.SetResult(responseMessage);
                }
            }
        }
        catch (WebSocketException)
        {
            // A conexão foi perdida, o socket será removido.
        }
        finally
        {
            // Garante que o socket falho seja removido do cache
            if (_sockets.TryRemove(url, out var staleSocket))
            {
                staleSocket.Dispose();
            }
        }
    }

    public void Dispose()
    {
        foreach (var socket in _sockets.Values)
        {
            socket.Dispose();
        }
    }
}