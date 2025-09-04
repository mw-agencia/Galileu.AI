using Galileu.ModelBase.Services;
using Microsoft.AspNetCore.Mvc;

namespace Galileu.ModelBase.Controllers;

[ApiController]
[Route("api/[controller]")]
public class MarkovController : ControllerBase
{
    private readonly MarkovService _svc;

    public MarkovController(MarkovService svc)
    {
        _svc = svc;
    }

    // Treina dataset em chunks
    // POST /api/markov/train-chunks?datasetPath=/absolute/path.txt&chunkSize=5000&epochs=3
    [HttpPost("train-chunks")]
    public async Task<IActionResult> TrainChunks([FromQuery] string datasetPath,
        [FromQuery] int chunkSize = 5000,
        [FromQuery] int epochs = 1,
        [FromQuery] bool shuffle = true,
        [FromQuery] bool saveAfterEachChunk = false)
    {
        if (string.IsNullOrWhiteSpace(datasetPath))
            return BadRequest("datasetPath query parameter is required (server-side file path).");
        try
        {
            await _svc.TrainWithChunksAsync(datasetPath, chunkSize, epochs, shuffle, saveAfterEachChunk);
            return Ok(new { message = "training completed", datasetPath, chunkSize, epochs });
        }
        catch (Exception ex)
        {
            return BadRequest(new { error = ex.Message });
        }
    }

    // Geração
    // GET /api/markov/generate?startToken=olá&maxSteps=20
    [HttpGet("generate")]
    public IActionResult Generate([FromQuery] string startToken = "<START>", [FromQuery] int maxSteps = 20,
        [FromQuery] double temperature = 1.0)
    {
        Console.WriteLine($"User Input :{startToken}");
        var text = _svc.Generate(startToken, maxSteps, temperature);
        Console.WriteLine($"markov generate :{text}");
        return Ok(new { generated = text });
    }

    // Save model
    [HttpPost("save-model")]
    public async Task<IActionResult> SaveModel()
    {
        await _svc.SaveModelAsync();
        return Ok(new { message = "model saved" });
    }

    // Load model
    [HttpPost("load-model")]
    public async Task<IActionResult> LoadModel()
    {
        await _svc.LoadModelAsync();
        return Ok(new { message = "model loaded" });
    }


    [HttpPost("reset-model")]
    public IActionResult ResetModel()
    {
        Console.WriteLine("Model state and vocabulary have been reset.");
        return Ok(new { message = "Model state and vocabulary have been reset." });
    }

    // Vocab
    [HttpGet("vocab")]
    public IActionResult Vocab()
    {
        return Ok(_svc.GetVocabulary());
    }

    // Summarize
    [HttpPost("summarize")]
    public IActionResult Summarize([FromBody] SummarizeRequest req)
    {
        if (req == null || string.IsNullOrWhiteSpace(req.Text)) return BadRequest("body must have 'text'.");
        var summary = _svc.SummarizeText(req.Text, req.Ratio, req.MaxSentences);
        return Ok(new { summary });
    }

    public class SummarizeRequest
    {
        public string Text { get; set; } = "";
        public double Ratio { get; set; } = 0.2;
        public int MaxSentences { get; set; } = 0;
    }
}