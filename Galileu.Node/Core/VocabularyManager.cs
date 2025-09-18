using System.Text.RegularExpressions;

namespace Galileu.Node.Core;

public class VocabularyManager
{
    private readonly Dictionary<string, int> vocab;
    private readonly Dictionary<int, string> reverseVocab;
    public string VocabFilePath { get; } = Path.Combine(Environment.CurrentDirectory, "Dayson", "vocab.txt");

    public VocabularyManager()
    {
        vocab = new Dictionary<string, int>();
        reverseVocab = new Dictionary<int, string>();
    }

    public Dictionary<string, int> Vocab => vocab;
    public Dictionary<int, string> ReverseVocab => reverseVocab;
    public int VocabSize => vocab.Count;

    public int BuildVocabulary(string datasetPath)
    {
        var tokens = new HashSet<string>();

        if (File.Exists(VocabFilePath))
        {
            foreach (var line in File.ReadAllLines(VocabFilePath))
            {
                if (!string.IsNullOrEmpty(line.Trim())) tokens.Add(line.Trim());
            }
        }

        // --- CORREÇÃO: Tokenização Aprimorada com Regex ---
        // Este padrão é a chave da solução. Ele captura:
        // 1. (\p{L}+) - Sequências de uma ou mais letras (palavras).
        // 2. (\p{N}+) - Sequências de um ou mais números.
        // 3. ([.,!?;:'"/\-]) - Qualquer um dos caracteres especiais listados, individualmente.
        string pattern = @"(\p{L}+|\p{N}+|[.,!?;:'""/\-])";

        if (File.Exists(datasetPath))
        {
            var datasetText = File.ReadAllText(datasetPath);
            var matches = Regex.Matches(datasetText.ToLower(), pattern);
            foreach (Match match in matches)
            {
                tokens.Add(match.Value);
            }
        }
        
        // Adiciona tokens especiais se não existirem
        tokens.Add("<PAD>"); // Padding
        tokens.Add("<UNK>"); // Unknown token

        vocab.Clear();
        reverseVocab.Clear();
        
        // Garante que o diretório exista antes de escrever o arquivo
        var directory = Path.GetDirectoryName(VocabFilePath);
        if (directory != null && !Directory.Exists(directory))
        {
            Directory.CreateDirectory(directory);
        }

        using (var writer = new StreamWriter(VocabFilePath, false))
        {
            int index = 0;
            // Ordena os tokens para garantir um vocabulário consistente entre execuções
            foreach (var token in tokens.OrderBy(t => t))
            {
                writer.WriteLine(token);
                vocab[token] = index;
                reverseVocab[index] = token;
                index++;
            }
        }

        return tokens.Count;
    }

    public int LoadVocabulary()
    {
        vocab.Clear();
        reverseVocab.Clear();

        if (!File.Exists(VocabFilePath))
        {
            return 0;
        }

        int index = 0;
        foreach (var line in File.ReadAllLines(VocabFilePath))
        {
            string token = line.Trim();
            if (!string.IsNullOrEmpty(token))
            {
                vocab[token] = index;
                reverseVocab[index] = token;
                index++;
            }
        }

        return vocab.Count;
    }
}