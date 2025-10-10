namespace Galileu.Node.Brain;

public class TensorMetadata
{
    public int Id { get; set; }
    public long Offset { get; set; }
    public int[] Shape { get; set; }
    public long SizeBytes { get; set; }
    public bool InUse { get; set; }
}