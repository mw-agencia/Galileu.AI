// --- START OF FILE GenerativeController.cs (FINAL CORRECTED VERSION) ---

using Galileu.Node.Core;
using Galileu.Node.Models;
using Galileu.Node.Services;
using Microsoft.AspNetCore.Mvc;

namespace Galileu.Node.Controllers;

[ApiController]
[Route("api/[controller]")]
public class GenerativeController : ControllerBase
{
    private readonly GenerativeService _generativeService;
    public string modelSavePath = "";

    public GenerativeController(GenerativeService generativeService)
    {
        _generativeService = generativeService;
        // --- MUDANÇA: Toda a lógica de inicialização foi REMOVIDA do construtor ---
    }

    [HttpGet("status")]
    public IActionResult GetStatus()
    {
        if (_generativeService.IsModelLoaded)
        {
            return Ok("Serviço generativo pronto e modelo carregado.");
        }
        return Ok("Serviço generativo em execução, mas nenhum modelo foi carregado. Use o endpoint /trainer.");
    }

    [HttpPost("trainer")]
    public async Task<IActionResult> Trainer([FromForm] Trainer trainer)
    {
        if (!System.IO.File.Exists(trainer.datasetPath))
        {
            return BadRequest($"Arquivo de dataset não encontrado em: {trainer.datasetPath}");
        }

        // A lógica de vocabulário e configuração é melhor tratada dentro do serviço,
        // mas por enquanto podemos mantê-la aqui para configurar o serviço antes de treinar.
        var vocabManager = new VocabularyManager();
        int vocabSize = vocabManager.BuildVocabulary(trainer.datasetPath);
        if (vocabSize == 0)
        {
            return StatusCode(500, "Falha ao construir ou carregar o vocabulário.");
        }

        // Configura o serviço com os parâmetros corretos antes do treinamento
        _generativeService.Configure(
            inputSize: vocabSize,
            hiddenSize: 256, // Manter consistente
            outputSize: vocabSize,
            contextWindowSize: 5,
            Path.Combine(Environment.CurrentDirectory, "Dayson", "Dayson.json"),
            searchService: new MockSearchService()
        );
        
        await _generativeService.TrainModelAsync(trainer);
        return Ok("Treinamento concluído e modelo recarregado para inferência!");
    }

    [HttpPost("generate")]
    public async Task<IActionResult> Generate([FromBody] GenerateResponse generateResponse)
    {
        if (!_generativeService.IsModelLoaded)
        {
            return BadRequest("O modelo não está carregado. Treine um modelo primeiro usando o endpoint /trainer.");
        }

        var response = await _generativeService.GenerateAsync(generateResponse);
        return Ok(response);
    }
}