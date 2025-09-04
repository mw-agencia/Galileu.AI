using System.Text;
using System.Text.Json;
using System.Linq;

namespace Galileu.ModelBase.Services;

public class MarkovService
{
    private readonly object _lock = new();
    private Dictionary<string, Dictionary<string, double>> _transitionMatrix;
    private HashSet<string> _vocabulary;
    private readonly string _dataFolder;
    private readonly string _vocabPath;
    private readonly string _statePath;
    private readonly string _embeddingsPath;
    private readonly string _attentionWeightsPath;

    // MHA e Rede Neural: Campos
    private readonly int _dModel = 32; // Dimensão do embedding
    private readonly int _numHeads = 8; // Número de cabeças
    private Dictionary<string, double[]> _embeddings; // Embeddings por token
    private double[][,] _Wq, _Wk, _Wv; // Matrizes de pesos Q, K, V por cabeça
    private readonly double _biasThreshold = 0.05; // Reduzido para maior flexibilidade
    private readonly double _learningRate = 0.0001; // Reduzido para estabilidade
    private readonly double _l2Regularization = 0.02; // Aumentado
    private readonly int _headDim;

    public MarkovService()
    {
        _dataFolder = Path.Combine(AppContext.BaseDirectory, "Data");
        Directory.CreateDirectory(_dataFolder);
        _vocabPath = Path.Combine(_dataFolder, "vocab.txt");
        _statePath = Path.Combine(_dataFolder, "state.json");
        _embeddingsPath = Path.Combine(_dataFolder, "embeddings.json");
        _attentionWeightsPath = Path.Combine(_dataFolder, "attention_weights.json");

        _transitionMatrix = new Dictionary<string, Dictionary<string, double>>(StringComparer.Ordinal);
        _vocabulary = new HashSet<string>(StringComparer.Ordinal);
        _embeddings = new Dictionary<string, double[]>(StringComparer.Ordinal);
        _headDim = _dModel / _numHeads;

        // Inicializar pesos de atenção
        _Wq = new double[_numHeads][,];
        _Wk = new double[_numHeads][,];
        _Wv = new double[_numHeads][,];
        var rnd = new Random();
        for (int h = 0; h < _numHeads; h++)
        {
            _Wq[h] = new double[_dModel, _headDim];
            _Wk[h] = new double[_dModel, _headDim];
            _Wv[h] = new double[_dModel, _headDim];
            for (int i = 0; i < _dModel; i++)
                for (int j = 0; j < _headDim; j++)
                {
                    _Wq[h][i, j] = rnd.NextDouble() * 0.005 - 0.0025; // Escala ainda menor
                    _Wk[h][i, j] = rnd.NextDouble() * 0.005 - 0.0025;
                    _Wv[h][i, j] = rnd.NextDouble() * 0.005 - 0.0025;
                }
        }

        LoadVocabulary();
        LoadStateIfExists();
        LoadEmbeddingsIfExists();
        LoadAttentionWeightsIfExists();
    }

    #region Vocabulary persistence

    private void LoadVocabulary()
    {
        if (!File.Exists(_vocabPath)) return;
        foreach (var line in File.ReadAllLines(_vocabPath))
        {
            var token = line.Trim();
            if (!string.IsNullOrWhiteSpace(token))
                _vocabulary.Add(token);
        }
    }

    private void PersistToken(string token)
    {
        if (string.IsNullOrWhiteSpace(token)) return;
        lock (_lock)
        {
            if (_vocabulary.Add(token))
            {
                File.AppendAllText(_vocabPath, token + Environment.NewLine, Encoding.UTF8);
            }
        }
    }

    public IEnumerable<string> GetVocabulary() => _vocabulary.OrderBy(x => x);

    #endregion

    private void LoadStateIfExists()
    {
        if (!File.Exists(_statePath)) return;
        try
        {
            var json = File.ReadAllText(_statePath);
            var data = JsonSerializer.Deserialize<Dictionary<string, Dictionary<string, double>>>(json);
            if (data != null)
            {
                var matrix = new Dictionary<string, Dictionary<string, double>>(StringComparer.Ordinal);
                foreach (var kvp in data)
                {
                    matrix[kvp.Key] = new Dictionary<string, double>(kvp.Value, StringComparer.Ordinal);
                }
                _transitionMatrix = matrix;
            }
        }
        catch
        {
            // Ignorar JSON malformado
        }
    }

    private void LoadEmbeddingsIfExists()
    {
        if (!File.Exists(_embeddingsPath)) return;
        try
        {
            var json = File.ReadAllText(_embeddingsPath);
            var data = JsonSerializer.Deserialize<Dictionary<string, double[]>>(json);
            if (data != null)
            {
                _embeddings = new Dictionary<string, double[]>(data, StringComparer.Ordinal);
            }
        }
        catch
        {
            // Ignorar JSON malformado
        }
    }

    private void LoadAttentionWeightsIfExists()
    {
        if (!File.Exists(_attentionWeightsPath)) return;
        try
        {
            var json = File.ReadAllText(_attentionWeightsPath);
            var data = JsonSerializer.Deserialize<double[][][]>(json);
            if (data != null && data.Length == 3 && data[0].Length == _numHeads)
            {
                for (int h = 0; h < _numHeads; h++)
                {
                    _Wq[h] = new double[_dModel, _headDim];
                    _Wk[h] = new double[_dModel, _headDim];
                    _Wv[h] = new double[_dModel, _headDim];
                    for (int i = 0; i < _dModel; i++)
                        for (int j = 0; j < _headDim; j++)
                        {
                            _Wq[h][i, j] = data[0][h][i * _headDim + j];
                            _Wk[h][i, j] = data[1][h][i * _headDim + j];
                            _Wv[h][i, j] = data[2][h][i * _headDim + j];
                        }
                }
            }
        }
        catch
        {
            // Ignorar JSON malformado
        }
    }

    public async Task SaveModelAsync()
    {
        lock (_lock)
        {
            var options = new JsonSerializerOptions { WriteIndented = true };
            var json = JsonSerializer.Serialize(_transitionMatrix, options);
            File.WriteAllText(_statePath, json, Encoding.UTF8);
            json = JsonSerializer.Serialize(_embeddings, options);
            File.WriteAllText(_embeddingsPath, json, Encoding.UTF8);

            var attentionWeights = new double[3][][];
            attentionWeights[0] = new double[_numHeads][];
            attentionWeights[1] = new double[_numHeads][];
            attentionWeights[2] = new double[_numHeads][];
            for (int h = 0; h < _numHeads; h++)
            {
                attentionWeights[0][h] = new double[_dModel * _headDim];
                attentionWeights[1][h] = new double[_dModel * _headDim];
                attentionWeights[2][h] = new double[_dModel * _headDim];
                for (int i = 0; i < _dModel; i++)
                    for (int j = 0; j < _headDim; j++)
                    {
                        attentionWeights[0][h][i * _headDim + j] = _Wq[h][i, j];
                        attentionWeights[1][h][i * _headDim + j] = _Wk[h][i, j];
                        attentionWeights[2][h][i * _headDim + j] = _Wv[h][i, j];
                    }
            }
            json = JsonSerializer.Serialize(attentionWeights, options);
            File.WriteAllText(_attentionWeightsPath, json, Encoding.UTF8);
        }
        await Task.CompletedTask;
    }

    public async Task LoadModelAsync()
    {
        if (!File.Exists(_statePath)) throw new FileNotFoundException("Modelo não encontrado", _statePath);
        var json = await File.ReadAllTextAsync(_statePath, Encoding.UTF8);
        var data = JsonSerializer.Deserialize<Dictionary<string, Dictionary<string, double>>>(json);
        if (data != null)
        {
            var matrix = new Dictionary<string, Dictionary<string, double>>(StringComparer.Ordinal);
            foreach (var kvp in data)
            {
                matrix[kvp.Key] = new Dictionary<string, double>(kvp.Value, StringComparer.Ordinal);
            }
            lock (_lock)
            {
                _transitionMatrix = matrix;
            }
        }

        if (File.Exists(_embeddingsPath))
        {
            json = await File.ReadAllTextAsync(_embeddingsPath, Encoding.UTF8);
            var embData = JsonSerializer.Deserialize<Dictionary<string, double[]>>(json);
            if (embData != null)
            {
                lock (_lock)
                {
                    _embeddings = new Dictionary<string, double[]>(embData, StringComparer.Ordinal);
                }
            }
        }

        LoadAttentionWeightsIfExists();
    }

    private double[] GetEmbedding(string token)
    {
        if (!_embeddings.TryGetValue(token, out var emb))
        {
            emb = new double[_dModel];
            var rnd = new Random(token.GetHashCode());
            for (int i = 0; i < _dModel; i++) emb[i] = rnd.NextDouble() * 0.005 - 0.0025;
            _embeddings[token] = emb;
        }
        // Normalização segura
        double norm = Math.Sqrt(emb.Sum(x => x * x));
        if (norm < 1e-6) return emb; // Evitar divisão por zero
        return emb.Select(x => x / norm).ToArray();
    }

    private double[] ComputeMHAContext(List<string> sequence)
    {
        if (sequence.Count == 0) return new double[_dModel];
        int seqLen = Math.Min(sequence.Count, 50);
        double[,] embeds = new double[seqLen, _dModel];
        for (int i = 0; i < seqLen; i++)
        {
            var emb = GetEmbedding(sequence[sequence.Count - seqLen + i]);
            for (int j = 0; j < _dModel; j++) embeds[i, j] = emb[j];
        }

        double[,] context = new double[seqLen, _dModel];
        for (int h = 0; h < _numHeads; h++)
        {
            double[,] Q = MatrixMultiply(embeds, _Wq[h]);
            double[,] K = MatrixMultiply(embeds, _Wk[h]);
            double[,] V = MatrixMultiply(embeds, _Wv[h]);

            double[,] scores = new double[seqLen, seqLen];
            double scale = Math.Sqrt(_headDim);
            for (int i = 0; i < seqLen; i++)
                for (int j = 0; j < seqLen; j++)
                {
                    double dot = 0;
                    for (int d = 0; d < _headDim; d++) dot += Q[i, d] * K[j, d];
                    scores[i, j] = Math.Max(Math.Min(dot / scale, 50), -50); // Clipping mais agressivo
                }

            for (int i = 0; i < seqLen; i++)
            {
                double sumExp = 0;
                for (int j = 0; j < seqLen; j++) sumExp += Math.Exp(scores[i, j]);
                sumExp = Math.Max(sumExp, 1e-12);
                for (int j = 0; j < seqLen; j++) scores[i, j] = Math.Exp(scores[i, j]) / sumExp;
            }

            double[,] headOut = new double[seqLen, _headDim];
            for (int i = 0; i < seqLen; i++)
                for (int d = 0; d < _headDim; d++)
                {
                    double sum = 0;
                    for (int j = 0; j < seqLen; j++) sum += scores[i, j] * V[j, d];
                    headOut[i, d] = sum;
                }

            for (int i = 0; i < seqLen; i++)
                for (int d = 0; d < _headDim; d++)
                    context[i, h * _headDim + d] += headOut[i, d];
        }

        double[] finalContext = new double[_dModel];
        for (int d = 0; d < _dModel; d++)
        {
            double sum = 0;
            for (int i = 0; i < seqLen; i++) sum += context[i, d];
            finalContext[d] = seqLen > 0 ? sum / seqLen : 0;
        }
        // Normalização segura
        double norm = Math.Sqrt(finalContext.Sum(x => x * x));
        if (norm < 1e-6) return finalContext;
        return finalContext.Select(x => x / norm).ToArray();
    }

    private double[,] MatrixMultiply(double[,] a, double[,] b)
    {
        int m = a.GetLength(0), n = a.GetLength(1), p = b.GetLength(1);
        double[,] res = new double[m, p];
        for (int i = 0; i < m; i++)
            for (int j = 0; j < p; j++)
            {
                double sum = 0;
                for (int k = 0; k < n; k++) sum += a[i, k] * b[k, j];
                res[i, j] = double.IsNaN(sum) ? 0 : sum; // Evitar NaN
            }
        return res;
    }

    #region Chunk training with loss and backpropagation

    public async Task TrainWithChunksAsync(string datasetPath, int chunkSize = 5000, int epochs = 1, bool shuffleEachEpoch = true, bool saveAfterEachChunk = false)
    {
        if (!File.Exists(datasetPath)) throw new FileNotFoundException("Dataset não encontrado", datasetPath);
        var allText = await File.ReadAllTextAsync(datasetPath, Encoding.UTF8);
        var tokens = Tokenize(allText);
        if (tokens.Count < 2) return;

        chunkSize = Math.Min(Math.Max(tokens.Count / 10, 100), 1000);
        Console.WriteLine($"Tamanho do chunk ajustado para: {chunkSize} tokens");

        foreach (var t in tokens.Distinct()) PersistToken(t);

        var windows = new List<int>();
        for (int i = 0; i <= tokens.Count - 2; i += chunkSize)
            windows.Add(i);

        var rnd = new Random();

        for (int epoch = 1; epoch <= Math.Max(1, epochs); epoch++)
        {
            double epochLossSum = 0;
            int epochChunkCount = 0;

            var idxs = windows.ToList();
            if (shuffleEachEpoch) idxs = idxs.OrderBy(_ => rnd.Next()).ToList();

            foreach (var start in idxs)
            {
                int end = Math.Min(start + chunkSize, tokens.Count);
                var chunkTokens = tokens.GetRange(start, end - start);
                double chunkLoss = TrainChunk(chunkTokens);
                epochLossSum += chunkLoss;
                epochChunkCount++;

                Console.WriteLine($"[Epoch {epoch}] Chunk {epochChunkCount}/{idxs.Count} -> Loss: {chunkLoss:F6}");

                if (saveAfterEachChunk)
                {
                    await SaveModelAsync();
                }
            }

            double epochAvg = epochChunkCount > 0 ? epochLossSum / epochChunkCount : 0;
            Console.WriteLine($"=== [Epoch {epoch}] Média de Loss: {epochAvg:F6} ===");
        }
    }

    private double TrainChunk(List<string> chunkTokens)
    {
        double totalLoss = 0;
        int count = 0;

        lock (_lock)
        {
            for (int i = 0; i < chunkTokens.Count - 1; i++)
            {
                var current = chunkTokens[i];
                var next = chunkTokens[i + 1];

                // Forward: Computar contexto e escores
                List<string> sequence = chunkTokens.Take(i + 1).ToList();
                double[] context = ComputeMHAContext(sequence);
                if (!_transitionMatrix.TryGetValue(current, out var dict))
                {
                    dict = new Dictionary<string, double>(StringComparer.Ordinal);
                    _transitionMatrix[current] = dict;
                }

                // Garantir que o token alvo esteja na matriz
                if (!dict.ContainsKey(next)) dict[next] = 1e-6; // Inicializar com valor pequeno

                var keys = dict.Keys.ToArray();
                var scores = new double[keys.Length];
                var currEmb = GetEmbedding(current);
                for (int j = 0; j < keys.Length; j++)
                {
                    scores[j] = dict[keys[j]];
                    var keyEmb = GetEmbedding(keys[j]);
                    double mod = 0;
                    for (int d = 0; d < _dModel; d++) mod += context[d] * keyEmb[d];
                    scores[j] += mod;
                    if (scores[j] < _biasThreshold) scores[j] = double.NegativeInfinity;
                }

                // Softmax com Laplace smoothing
                double max = scores.Where(s => !double.IsNegativeInfinity(s)).DefaultIfEmpty(0).Max();
                var probs = scores.Select(s => double.IsNegativeInfinity(s) ? 1e-6 : Math.Exp(Math.Max(Math.Min(s - max, 50), -50)) + 1e-6).ToArray();
                double sum = probs.Sum();
                probs = probs.Select(p => p / sum).ToArray();

                // Loss: Entropia cruzada
                int targetIdx = Array.IndexOf(keys, next);
                double prob = targetIdx >= 0 ? probs[targetIdx] : 1e-6;
                double loss = -Math.Log(prob);
                if (double.IsNaN(loss) || double.IsInfinity(loss))
                {
                    Console.WriteLine($"[DEBUG] Loss inválida: prob={prob}, target={next}, scores={string.Join(",", scores.Take(5))}");
                    continue;
                }
                totalLoss += loss;
                count++;

                // Backpropagation
                if (targetIdx >= 0)
                {
                    var gradProbs = new double[probs.Length];
                    for (int j = 0; j < probs.Length; j++)
                        gradProbs[j] = probs[j] - (j == targetIdx ? 1.0 : 0.0);

                    // Clipping de gradientes
                    double gradNorm = Math.Sqrt(gradProbs.Sum(g => g * g));
                    if (gradNorm > 0.5) gradProbs = gradProbs.Select(g => g * 0.5 / gradNorm).ToArray();

                    // Atualizar embeddings com L2
                    for (int j = 0; j < keys.Length; j++)
                    {
                        if (gradProbs[j] != 0)
                        {
                            var keyEmb = GetEmbedding(keys[j]);
                            for (int d = 0; d < _dModel; d++)
                            {
                                double grad = gradProbs[j] * context[d];
                                if (!double.IsNaN(grad))
                                    keyEmb[d] -= _learningRate * (grad + _l2Regularization * keyEmb[d]);
                            }
                        }
                    }

                    // Atualizar transitionMatrix
                    for (int j = 0; j < keys.Length; j++)
                    {
                        if (gradProbs[j] != 0)
                        {
                            double grad = gradProbs[j];
                            if (!double.IsNaN(grad))
                            {
                                dict[keys[j]] -= _learningRate * grad;
                                if (dict[keys[j]] < 0) dict[keys[j]] = 1e-6;
                            }
                        }
                    }

                    // Atualizar pesos de atenção
                    for (int h = 0; h < _numHeads; h++)
                    {
                        for (int d = 0; d < _dModel; d++)
                            for (int k = 0; k < _headDim; k++)
                            {
                                double grad = gradProbs[targetIdx] * currEmb[d];
                                if (!double.IsNaN(grad))
                                {
                                    _Wq[h][d, k] -= _learningRate * (grad + _l2Regularization * _Wq[h][d, k]);
                                    _Wk[h][d, k] -= _learningRate * (grad + _l2Regularization * _Wk[h][d, k]);
                                    _Wv[h][d, k] -= _learningRate * (grad + _l2Regularization * _Wv[h][d, k]);
                                }
                            }
                    }
                }

                // Contagem simples para compatibilidade
                dict[next] = dict.GetValueOrDefault(next, 0) + 1.0;
            }
        }

        return count > 0 ? totalLoss / count : 0.0;
    }

    #endregion

    #region Generation

    public string Generate(string startToken, int maxSteps = 20, double temperature = 1.0)
    {
        if (string.IsNullOrWhiteSpace(startToken)) startToken = "<START>";
        var sb = new StringBuilder();
        sb.Append(startToken);
        var rnd = new Random();
        string current = startToken;
        List<string> sequence = new List<string> { "<START>", startToken };

        for (int step = 0; step < maxSteps; step++)
        {
            if (!_transitionMatrix.TryGetValue(current, out var dict) || dict.Count == 0) break;

            double[] context = ComputeMHAContext(sequence);
            var next = SampleNext(dict, rnd, temperature, context);
            if (next == "<STOP>" || string.IsNullOrEmpty(next)) break;

            sb.Append(' ').Append(next);
            current = next;
            sequence.Add(next);
        }

        return sb.ToString();
    }

    private string SampleNext(Dictionary<string, double> dict, Random rnd, double temperature, double[] context)
    {
        var keys = dict.Keys.ToArray();
        var scores = keys.Select(k => dict[k]).ToArray();

        for (int i = 0; i < keys.Length; i++)
        {
            var keyEmb = GetEmbedding(keys[i]);
            double mod = 0;
            for (int d = 0; d < _dModel; d++) mod += context[d] * keyEmb[d];
            scores[i] += mod;
            if (scores[i] < _biasThreshold) scores[i] = double.NegativeInfinity;
        }

        double max = scores.Where(s => !double.IsNegativeInfinity(s)).DefaultIfEmpty(0).Max();
        var exps = scores.Select(s => double.IsNegativeInfinity(s) ? 1e-6 : Math.Exp(Math.Max(Math.Min(s - max, 50), -50)) + 1e-6).ToArray();
        double sum = exps.Sum();
        exps = exps.Select(e => e / sum).ToArray();

        double r = rnd.NextDouble();
        double cum = 0;
        for (int i = 0; i < exps.Length; i++)
        {
            cum += exps[i];
            if (r <= cum) return keys[i];
        }
        return keys.LastOrDefault() ?? "<STOP>";
    }

    #endregion

    #region Summarization (TextRank-lite)

    private static readonly System.Text.RegularExpressions.Regex SentenceSplitter =
        new(@"(?<=[\.!\?])\s+");

    public string SummarizeText(string text, double ratio = 0.2, int maxSentences = 0)
    {
        if (string.IsNullOrWhiteSpace(text)) return string.Empty;
        var sentences = SentenceSplitter.Split(text.Trim())
            .Select(s => s.Trim()).Where(s => !string.IsNullOrWhiteSpace(s)).ToList();

        int n = sentences.Count;
        if (n == 0) return string.Empty;
        if (n == 1) return sentences[0];

        var tokenized = sentences.Select(s => TokenizeWords(s)).ToList();
        var sim = new double[n, n];
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                sim[i, j] = i == j ? 0 : SentenceSimilarity(tokenized[i], tokenized[j]);

        var scores = Enumerable.Repeat(1.0, n).ToArray();
        const double damping = 0.85;
        const double eps = 1e-4;
        for (int iter = 0; iter < 100; iter++)
        {
            var next = new double[n];
            double maxDiff = 0;
            for (int i = 0; i < n; i++)
            {
                double sum = 0;
                for (int j = 0; j < n; j++)
                {
                    if (sim[j, i] <= 0) continue;
                    double outW = 0;
                    for (int k = 0; k < n; k++) outW += sim[j, k];
                    if (outW == 0) continue;
                    sum += (sim[j, i] / outW) * scores[j];
                }
                next[i] = (1 - damping) + damping * sum;
                maxDiff = Math.Max(maxDiff, Math.Abs(next[i] - scores[i]));
            }
            Array.Copy(next, scores, n);
            if (maxDiff < eps) break;
        }

        int selectCount = maxSentences > 0 ? Math.Min(maxSentences, n) : Math.Max(1, (int)Math.Ceiling(n * Math.Clamp(ratio, 0.05, 1.0)));
        var chosen = Enumerable.Range(0, n)
            .Select(i => new { i, score = scores[i] })
            .OrderByDescending(x => x.score)
            .Take(selectCount)
            .Select(x => x.i)
            .OrderBy(i => i)
            .ToList();

        return string.Join(" ", chosen.Select(i => sentences[i]));
    }

    private static List<string> TokenizeWords(string sentence)
    {
        var punct = ".,;:!?\"'()[]{}".ToCharArray();
        return sentence.ToLowerInvariant()
            .Replace("\n", " ")
            .Split(' ', StringSplitOptions.RemoveEmptyEntries)
            .Select(t => t.Trim(punct))
            .Where(t => !string.IsNullOrWhiteSpace(t))
            .ToList();
    }

    private static double SentenceSimilarity(List<string> a, List<string> b)
    {
        if (a.Count == 0 || b.Count == 0) return 0.0;
        var all = new HashSet<string>(a);
        all.UnionWith(b);
        var va = new double[all.Count];
        var vb = new double[all.Count];
        int idx = 0;
        foreach (var w in all)
        {
            va[idx] = a.Count(x => x == w);
            vb[idx] = b.Count(x => x == w);
            idx++;
        }
        double dot = 0, na = 0, nb = 0;
        for (int i = 0; i < va.Length; i++)
        {
            dot += va[i] * vb[i];
            na += va[i] * va[i];
            nb += vb[i] * vb[i];
        }
        if (na == 0 || nb == 0) return 0;
        return dot / (Math.Sqrt(na) * Math.Sqrt(nb));
    }

    #endregion

    #region Helpers

    private static List<string> Tokenize(string text)
    {
        var punct = ".,;:!?\"'()[]{}".ToCharArray();
        return text.ToLowerInvariant()
            .Replace("\n", " ")
            .Split(' ', StringSplitOptions.RemoveEmptyEntries)
            .Select(t => t.Trim(punct))
            .Where(t => !string.IsNullOrWhiteSpace(t))
            .ToList();
    }

    public void ResetModel()
    {
        lock (_lock)
        {
            _transitionMatrix = new Dictionary<string, Dictionary<string, double>>(StringComparer.Ordinal);
            _vocabulary = new HashSet<string>(StringComparer.Ordinal);
            _embeddings = new Dictionary<string, double[]>(StringComparer.Ordinal);
            if (File.Exists(_vocabPath)) File.Delete(_vocabPath);
            if (File.Exists(_statePath)) File.Delete(_statePath);
            if (File.Exists(_embeddingsPath)) File.Delete(_embeddingsPath);
            if (File.Exists(_attentionWeightsPath)) File.Delete(_attentionWeightsPath);
        }
    }

    #endregion
}