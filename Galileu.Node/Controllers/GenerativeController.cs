// --- START OF FILE GenerativeController.cs ---

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

    public GenerativeController(GenerativeService generativeService)
    {
        _generativeService = generativeService;
    }

    [HttpGet]
    public IActionResult Get()
    {
        return Ok("GenerativeController está pronto. Chame o endpoint /trainer para iniciar.");
    }

    // O endpoint build-vocabulary foi removido, pois sua lógica agora está no /trainer.

    [HttpPost("trainer")]
    public async Task<IActionResult> Trainer([FromForm] Trainer trainer)
    {
        try
        {
            if (!System.IO.File.Exists(trainer.datasetPath))
            {
                return BadRequest($"Arquivo de dataset não encontrado em: {trainer.datasetPath}");
            }

            var vocabManager = new VocabularyManager();
            if (!System.IO.File.Exists(vocabManager.VocabFilePath))
            {
                Console.WriteLine("[GenerativeController] Vocabulário não encontrado. Criando a partir do dataset fornecido...");
                vocabManager.BuildVocabulary(trainer.datasetPath);
            }

            // 2. Carrega o vocabulário para obter seu tamanho.
            int vocabSize = vocabManager.LoadVocabulary();
            if (vocabSize == 0)
            {
                return StatusCode(500, "Falha ao construir ou carregar o vocabulário. O dataset pode estar vazio.");
            }

            // 3. Configura o serviço com os parâmetros corretos e dinâmicos.
            Console.WriteLine($"[GenerativeController] Vocabulário pronto. Tamanho: {vocabSize} tokens. Configurando o serviço...");
            int contextWindowSize = 5;
            int inputSize = contextWindowSize * vocabSize;
            int hiddenSize = 128;
            int outputSize = vocabSize;
            var modelSavePath = Path.Combine(Environment.CurrentDirectory, "Dayson", "Dayson.json");

            _generativeService.Configure(
                inputSize,
                hiddenSize,
                outputSize,
                contextWindowSize,
                modelSavePath,
                new MockSearchService()
            );

            // 4. Inicia o treinamento.
            await _generativeService.TrainModelAsync(trainer);
            return Ok("Treinamento concluído!");
        }
        catch (Exception ex)
        {
            return StatusCode(500, $"Um erro inesperado ocorreu durante o treinamento: {ex.Message}");
        }
    }

    [HttpPost("generate")]
    public async Task<IActionResult> Generate([FromBody] GenerateResponse generateResponse)
    {
        if (!_generativeService.IsConfigured)
        {
            if (!_generativeService.TryLoadConfigurationFromModel())
            {
                return BadRequest("O serviço não está configurado. Treine um modelo primeiro usando o endpoint /trainer.");
            }
        }
        var response = await _generativeService.GenerateAsync(generateResponse);
        return Ok(response);
    }
}