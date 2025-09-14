using MongoDB.Bson;
using MongoDB.Bson.Serialization.Attributes;

namespace Galileu.Models;

public class TransactionDocument
{
    [BsonId]
    [BsonRepresentation(BsonType.ObjectId)]
    public string Id { get; set; }

    [BsonElement("fromAddress")]
    public string FromAddress { get; set; }

    [BsonElement("toAddress")]
    public string ToAddress { get; set; }

    [BsonElement("amount")]
    [BsonRepresentation(BsonType.Decimal128)] // Usar Decimal128 para precisão financeira
    public decimal Amount { get; set; }

    [BsonElement("timestamp")]
    public DateTime Timestamp { get; set; }

    [BsonElement("notes")]
    public string? Notes { get; set; }

    [BsonElement("hash")]
    public string Hash { get; set; }
}