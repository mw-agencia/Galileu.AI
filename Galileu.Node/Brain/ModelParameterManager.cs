using Galileu.Node.Interfaces;
using System;
using System.Collections.Generic;
using System.IO;
using System.IO.MemoryMappedFiles;
using System.Linq;

namespace Galileu.Node.Brain;

/// <summary>
/// Gerencia TODOS os parâmetros do modelo (pesos, biases, estados do Adam) em disco.
/// Carrega parâmetros para a GPU/CPU sob demanda para cálculos e os descarrega imediatamente.
/// Mantém apenas metadados leves em RAM, garantindo um consumo de memória mínimo.
/// </summary>
public class ModelParameterManager : IDisposable
{
    private readonly IMathEngine _mathEngine;
    private readonly MemoryMappedFile _mmf;
    private readonly MemoryMappedViewAccessor _accessor;
    private readonly string _filePath;

    // Metadados em RAM (muito pequenos)
    private readonly Dictionary<string, (long offset, int[] shape, int length)> _paramInfo;
    private readonly Dictionary<string, (long m_offset, long v_offset, int t)> _adamInfo;

    private const long FILE_SIZE_BYTES = 12L * 1024 * 1024 * 1024; // 4 GB de espaço para parâmetros e buffers Adam

    public ModelParameterManager(IMathEngine mathEngine, Dictionary<string, int[]> shapes)
    {
        _mathEngine = mathEngine;
        _paramInfo = new Dictionary<string, (long, int[], int)>();
        _adamInfo = new Dictionary<string, (long, long, int)>();
        
        var dir = Path.Combine(Environment.CurrentDirectory, "Dayson", "Params");
        Directory.CreateDirectory(dir);
        _filePath = Path.Combine(dir, $"params_{Guid.NewGuid()}.bin");

        _mmf = MemoryMappedFile.CreateFromFile(_filePath, FileMode.Create, null, FILE_SIZE_BYTES);
        _accessor = _mmf.CreateViewAccessor();

        InitializeParameters(shapes);
        Console.WriteLine($"[ParamManager] Parâmetros do modelo alocados em disco: {_filePath}");
    }
    
    public ModelParameterManager(IMathEngine mathEngine, string sourceWeightsPath, Dictionary<string, int[]> shapes)
    {
        _mathEngine = mathEngine;
        _paramInfo = new Dictionary<string, (long, int[], int)>();
        _adamInfo = new Dictionary<string, (long, long, int)>();

        var dir = Path.Combine(Environment.CurrentDirectory, "Dayson", "Params");
        Directory.CreateDirectory(dir);
        _filePath = Path.Combine(dir, $"params_{Guid.NewGuid()}.bin");

        _mmf = MemoryMappedFile.CreateFromFile(_filePath, FileMode.Create, null, FILE_SIZE_BYTES);
        _accessor = _mmf.CreateViewAccessor();

        // Copia os dados do arquivo de origem para o MMF
        using (var sourceStream = File.OpenRead(sourceWeightsPath))
        using (var destStream = _mmf.CreateViewStream())
        {
            sourceStream.CopyTo(destStream);
        }
        
        // Reconstrói os metadados a partir das formas
        RebuildMetadata(shapes);
        Console.WriteLine($"[ParamManager] Parâmetros do modelo carregados de '{sourceWeightsPath}' para o disco.");
    }

    private void RebuildMetadata(Dictionary<string, int[]> shapes)
    {
        long currentOffset = 0;
        foreach (var pair in shapes)
        {
            string name = pair.Key;
            int[] shape = pair.Value;
            int length = shape.Aggregate(1, (a, b) => a * b);
            long sizeBytes = (long)length * sizeof(double);

            long paramOffset = currentOffset;
            long mOffset = paramOffset + sizeBytes;
            long vOffset = mOffset + sizeBytes;

            _paramInfo[name] = (paramOffset, shape, length);
            // Ao carregar, o estado do Adam também é carregado. Assumimos que o 't' é lido do arquivo.
            // Para simplificar, resetamos o estado do Adam ao carregar.
            _adamInfo[name] = (mOffset, vOffset, 0); 

            currentOffset = vOffset + sizeBytes;
        }
    }

    /// <summary>
    /// Aloca espaço no arquivo mapeado em memória para todos os parâmetros e buffers do otimizador,
    /// e inicializa os pesos.
    /// </summary>
    private void InitializeParameters(Dictionary<string, int[]> shapes)
    {
        long currentOffset = 0;
        var rand = new Random();

        foreach (var pair in shapes)
        {
            string name = pair.Key;
            int[] shape = pair.Value;
            int length = shape.Aggregate(1, (a, b) => a * b);
            long sizeBytes = (long)length * sizeof(double);

            // Aloca espaço para: Parâmetro, Momento M, Momento V
            if (currentOffset + sizeBytes * 3 > FILE_SIZE_BYTES)
                throw new OutOfMemoryException("Espaço do arquivo de parâmetros insuficiente. Aumente FILE_SIZE_BYTES.");

            long paramOffset = currentOffset;
            long mOffset = paramOffset + sizeBytes;
            long vOffset = mOffset + sizeBytes;

            _paramInfo[name] = (paramOffset, shape, length);
            _adamInfo[name] = (mOffset, vOffset, 0); // Timestep inicial é 0

            // Inicializa parâmetro com valores aleatórios (Xavier/Glorot)
            double limit = Math.Sqrt(6.0 / (shape[0] + (shape.Length > 1 ? shape[1] : 1)));
            var paramData = new double[length];
            for (int i = 0; i < length; i++)
            {
                paramData[i] = (rand.NextDouble() * 2 - 1) * limit;
            }
            _accessor.WriteArray(paramOffset, paramData, 0, length);

            // Zera os buffers do Adam no disco
            _accessor.WriteArray(mOffset, new double[length], 0, length);
            _accessor.WriteArray(vOffset, new double[length], 0, length);

            currentOffset = vOffset + sizeBytes;
        }
    }

    /// <summary>
    /// Carrega um parâmetro do disco para um tensor computacional (GPU/CPU).
    /// O tensor retornado DEVE ser descartado (usado em um bloco 'using') após o uso.
    /// </summary>
    public IMathTensor GetParameter(string name)
    {
        var (offset, shape, length) = _paramInfo[name];
        var data = new double[length];
        _accessor.ReadArray(offset, data, 0, length);
        return _mathEngine.CreateTensor(data, shape);
    }
    
    /// <summary>
    /// Aplica o gradiente a um parâmetro usando Adam e salva tudo de volta no disco.
    /// Esta operação é feita "out-of-core", carregando e salvando dados conforme necessário.
    /// </summary>
    public void UpdateParameter(string name, IMathTensor gradient, double learningRate)
    {
        var (paramOffset, _, paramLength) = _paramInfo[name];
        var (mOffset, vOffset, t) = _adamInfo[name];

        // Carrega os dados necessários do disco para a RAM
        var parameters = new double[paramLength];
        var m_buffer = new double[paramLength];
        var v_buffer = new double[paramLength];
        var gradData = gradient.ToCpuTensor().GetData();

        _accessor.ReadArray(paramOffset, parameters, 0, paramLength);
        _accessor.ReadArray(mOffset, m_buffer, 0, paramLength);
        _accessor.ReadArray(vOffset, v_buffer, 0, paramLength);

        // Executa o Adam Optimizer (que agora é stateless)
        AdamOptimizer.UpdateParameters(parameters, gradData, m_buffer, v_buffer, ref t, learningRate);

        // Salva os resultados (parâmetros atualizados e buffers do otimizador) de volta no disco
        _accessor.WriteArray(paramOffset, parameters, 0, paramLength);
        _accessor.WriteArray(mOffset, m_buffer, 0, paramLength);
        _accessor.WriteArray(vOffset, v_buffer, 0, paramLength);

        // Atualiza o metadado do timestep na RAM
        _adamInfo[name] = (mOffset, vOffset, t);
    }

    /// <summary>
    /// Salva o estado atual do arquivo mapeado em memória para um arquivo permanente no disco.
    /// </summary>
    /// <param name="destinationPath">O caminho completo do arquivo de destino (ex: 'modelo.bin').</param>
    public void SaveToFile(string destinationPath)
    {
        // 1. Flush: Garante que todas as escritas pendentes na memória sejam gravadas no arquivo MMF.
        //    Isso é crucial para garantir que os dados no disco estejam 100% atualizados.
        _accessor.Flush();

        // 2. Cópia de Arquivo: Cria uma cópia direta do arquivo MMF de trabalho
        //    para o local de destino permanente. Esta é a maneira mais eficiente
        //    de persistir o estado, pois evita carregar os dados para a RAM.
        try
        {
            File.Copy(_filePath, destinationPath, true); // O 'true' permite sobrescrever um arquivo de checkpoint anterior.
            Console.WriteLine($"[ParamManager] Estado dos parâmetros salvo com sucesso em '{destinationPath}'.");
        }
        catch (IOException ex)
        {
            Console.WriteLine($"[ParamManager] ERRO: Falha ao salvar o arquivo de parâmetros em '{destinationPath}'. {ex.Message}");
        }
    }

    public void Dispose()
    {
        _accessor?.Dispose();
        _mmf?.Dispose();
        try { if (File.Exists(_filePath)) File.Delete(_filePath); } catch { }
    }
}