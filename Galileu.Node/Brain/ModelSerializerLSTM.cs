using Galileu.Node.Core;
using Galileu.Node.Interfaces;
using System;
using System.IO;
using System.Text.Json; // Adicionado para ler os metadados do modelo

namespace Galileu.Node.Brain;

public class ModelSerializerLSTM
{
    /// <summary>
    /// Delega a lógica de salvamento para o método SaveModel da instância do modelo.
    /// Esta abordagem permanece correta e é bom design.
    /// </summary>
    public static void SaveModel(GenerativeNeuralNetworkLSTM model, string filePath)
    {
        if (model == null)
        {
            throw new ArgumentNullException(nameof(model));
        }
        model.SaveModel(filePath);
    }

    /// <summary>
    /// Carrega um modelo generativo completo seguindo a arquitetura "Out-of-Core".
    /// </summary>
    /// <param name="filePath">O caminho para o arquivo de configuração .json do modelo.</param>
    /// <param name="mathEngine">A engine de computação a ser usada.</param>
    /// <returns>Uma instância de GenerativeNeuralNetworkLSTM ou null se o carregamento falhar.</returns>
    public static GenerativeNeuralNetworkLSTM? LoadModel(string filePath, IMathEngine mathEngine)
    {
        try
        {
            // 1. Valida a existência dos arquivos necessários (.json e .bin)
            if (!File.Exists(filePath))
            {
                Console.WriteLine($"[Serializer] Erro: Arquivo de configuração do modelo não encontrado: {filePath}");
                return null;
            }
            string weightsPath = Path.ChangeExtension(filePath, ".bin");
            if (!File.Exists(weightsPath))
            {
                Console.WriteLine($"[Serializer] Erro: Arquivo de pesos do modelo não encontrado: {weightsPath}");
                return null;
            }

            // 2. Carrega o vocabulário.
            var vocabManager = new VocabularyManager();
            int vocabSize = vocabManager.LoadVocabulary();
            if (vocabSize == 0)
            {
                Console.WriteLine("[Serializer] Erro: Vocabulário não pôde ser carregado. O arquivo 'vocab.txt' está ausente ou vazio.");
                return null;
            }

            // 3. Carrega os metadados do modelo (.json) para validação ANTES de criar o objeto.
            string jsonString = File.ReadAllText(filePath);
            var modelData = JsonSerializer.Deserialize<NeuralNetworkModelDataEmbeddingLSTM>(jsonString);
            if (modelData == null)
            {
                Console.WriteLine($"[Serializer] Erro: Falha ao desserializar os metadados do modelo de '{filePath}'.");
                return null;
            }

            // 4. Valida a consistência entre o vocabulário carregado e os metadados do modelo.
            if (vocabSize != modelData.OutputSize)
            {
                Console.WriteLine(
                    $"[Serializer] Erro de Inconsistência: Tamanho do vocabulário ({vocabSize}) não corresponde ao OutputSize do modelo ({modelData.OutputSize}).");
                return null;
            }

            // 5. Instancia o modelo generativo usando o construtor correto "Out-of-Core".
            // Este construtor passará os caminhos de arquivo para as classes base, que por sua vez
            // inicializarão o ModelParameterManager com os pesos do disco.
            Console.WriteLine("[Serializer] Metadados e vocabulário validados. Instanciando modelo Out-of-Core...");
            return new GenerativeNeuralNetworkLSTM(filePath, vocabManager, new MockSearchService(), mathEngine);
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[Serializer] Erro crítico ao carregar o modelo LSTM generativo: {ex.Message}");
            return null;
        }
    }
}