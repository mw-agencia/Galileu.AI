namespace Galileu.Node.Core;

public class VocabularyManager
{
    private readonly Dictionary<string, int> vocab;
    private readonly Dictionary<int, string> reverseVocab;
    private readonly string vocabFile = "vocab.txt";

    public VocabularyManager()
    {
        vocab = new Dictionary<string, int>();
        reverseVocab = new Dictionary<int, string>();
    }

    /// <summary>
    /// Obtém o dicionário de vocabulário (token para índice).
    /// </summary>
    public Dictionary<string, int> Vocab => vocab;

    /// <summary>
    /// Obtém o dicionário reverso de vocabulário (índice para token).
    /// </summary>
    public Dictionary<int, string> ReverseVocab => reverseVocab;

    /// <summary>
    /// Obtém o tamanho do vocabulário.
    /// </summary>
    public int VocabSize => vocab.Count;

    /// <summary>
    /// Constrói o vocabulário a partir de um arquivo de dataset e salva em vocab.txt.
    /// </summary>
    /// <param name="datasetPath">Caminho do arquivo de dataset.</param>
    /// <returns>Tamanho do vocabulário construído.</returns>
    public int BuildVocabulary(string datasetPath)
    {
        HashSet<string> tokens = new HashSet<string>();

        // Carrega tokens existentes de vocab.txt, se disponível
        if (File.Exists(vocabFile))
        {
            foreach (var line in File.ReadAllLines(vocabFile))
            {
                string token = line.Trim();
                if (!string.IsNullOrEmpty(token))
                {
                    tokens.Add(token);
                }
            }
        }

        // Extrai novos tokens do dataset
        if (File.Exists(datasetPath))
        {
            foreach (var line in File.ReadAllLines(datasetPath))
            {
                var newTokens = line.Split(new[] { ' ' }, StringSplitOptions.RemoveEmptyEntries)
                                    .Select(t => t.Trim());
                foreach (var token in newTokens)
                {
                    tokens.Add(token);
                }
            }
        }

        // Salva o vocabulário em vocab.txt e mapeia tokens para índices
        vocab.Clear();
        reverseVocab.Clear();
        using (var writer = new StreamWriter(vocabFile, false))
        {
            int index = 0;
            foreach (var token in tokens)
            {
                writer.WriteLine(token);
                vocab[token] = index;
                reverseVocab[index] = token;
                index++;
            }
        }

        return tokens.Count;
    }

    /// <summary>
    /// Carrega o vocabulário de um arquivo vocab.txt existente.
    /// </summary>
    /// <returns>Tamanho do vocabulário carregado.</returns>
    public int LoadVocabulary()
    {
        vocab.Clear();
        reverseVocab.Clear();

        if (!File.Exists(vocabFile))
        {
            throw new FileNotFoundException($"Arquivo de vocabulário não encontrado em: {vocabFile}");
        }

        int index = 0;
        foreach (var line in File.ReadAllLines(vocabFile))
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