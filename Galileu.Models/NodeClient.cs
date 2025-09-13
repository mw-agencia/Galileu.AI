using System.Collections.Concurrent;
using System.Net.WebSockets;
using System.Text;
using System.Text.Json;

namespace Galileu.Models;


public class NodeClient : IDisposable
{
    private readonly ConcurrentDictionary<string, ClientWebSocket> _sockets = new();
    private readonly ConcurrentDictionary<Guid, TaskCompletionSource<Message>> _pendingRequests = new();

    public async Task<TResponse> SendRequestAsync<TResponse>(string targetAddress, Message request, CancellationToken token) where TResponse : Message
    {
        var wsUrl = targetAddress.Replace("http://", "ws://").Replace("https://", "wss://") + "/ws";
        var ws = _sockets.GetOrAdd(wsUrl, _ => new ClientWebSocket());

        if (ws.State != WebSocketState.Open)
        {
            await ws.ConnectAsync(new Uri(wsUrl), token);
            _ = ReceiveLoop(ws, wsUrl); // Inicia o loop de recebimento em background
        }
        
        var tcs = new TaskCompletionSource<Message>();
        _pendingRequests[request.CorrelationId] = tcs;

        var jsonMessage = JsonSerializer.Serialize(request, request.GetType());
        var buffer = Encoding.UTF8.GetBytes(jsonMessage);
        await ws.SendAsync(new ArraySegment<byte>(buffer), WebSocketMessageType.Text, true, token);

        var responseMessage = await tcs.Task.WaitAsync(TimeSpan.FromSeconds(10), token);
        return (TResponse)responseMessage;
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
        catch (Exception)
        {
            // Ocorreu um erro, remover o socket do pool.
            _sockets.TryRemove(url, out _);
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