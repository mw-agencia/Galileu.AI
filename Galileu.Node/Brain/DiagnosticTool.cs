using System;
using System.Diagnostics;
using System.IO;
using System.Linq;
using Galileu.Node.Brain;
using Galileu.Node.Interfaces;

namespace Galileu.Node.Tools;

/// <summary>
/// Ferramenta de diagnóstico para detectar vazamentos de memória durante treinamento.
/// Executa testes sintéticos e valida a arquitetura Zero Cache RAM.
/// </summary>
public class DiagnosticTool
{
    private readonly IMathEngine _mathEngine;
    private readonly Process _process;

    public DiagnosticTool(IMathEngine mathEngine)
    {
        _mathEngine = mathEngine;
        _process = Process.GetCurrentProcess();
    }

    /// <summary>
    /// Executa bateria completa de testes.
    /// </summary>
    public void RunFullDiagnostic()
    {
        Console.WriteLine("\n" + new string('═', 70));
        Console.WriteLine("FERRAMENTA DE DIAGNÓSTICO - ZERO CACHE RAM");
        Console.WriteLine(new string('═', 70));

        TestDiskOnlyCacheManager();
        TestTensorPoolStrict();
        TestForwardBackwardCycle();
        TestMemoryStability();
        
        Console.WriteLine("\n" + new string('═', 70));
        Console.WriteLine("DIAGNÓSTICO CONCLUÍDO");
        Console.WriteLine(new string('═', 70) + "\n");
    }

    /// <summary>
    /// Teste 1: Valida DiskOnlyCacheManager.
    /// </summary>
    private void TestDiskOnlyCacheManager()
    {
        Console.WriteLine("\n[Teste 1] DiskOnlyCacheManager - Gravação e Liberação");
        Console.WriteLine(new string('-', 70));

        Console.WriteLine("\n[Teste 1] DiskOnlyCacheManager - Gravação e Liberação");
        Console.WriteLine(new string('-', 70));

        long memoryBefore = GetMemoryMB();
        Console.WriteLine($"RAM inicial: {memoryBefore}MB");

        try
        {
            using var cacheManager = new DiskOnlyCacheManager(_mathEngine, embeddingSize: 128, hiddenSize: 256);
            
            // Simula 100 timesteps sendo cacheados
            Console.WriteLine("Cacheando 100 timesteps...");
            for (int t = 0; t < 100; t++)
            {
                var stepCache = new LstmStepCache
                {
                    Input = _mathEngine.CreateTensor(new[] { 1, 128 }),
                    HiddenPrev = _mathEngine.CreateTensor(new[] { 1, 256 }),
                    CellPrev = _mathEngine.CreateTensor(new[] { 1, 256 }),
                    ForgetGate = _mathEngine.CreateTensor(new[] { 1, 256 }),
                    InputGate = _mathEngine.CreateTensor(new[] { 1, 256 }),
                    CellCandidate = _mathEngine.CreateTensor(new[] { 1, 256 }),
                    OutputGate = _mathEngine.CreateTensor(new[] { 1, 256 }),
                    CellNext = _mathEngine.CreateTensor(new[] { 1, 256 }),
                    TanhCellNext = _mathEngine.CreateTensor(new[] { 1, 256 }),
                    HiddenNext = _mathEngine.CreateTensor(new[] { 1, 256 })
                };

                // CRÍTICO: CacheStep deve fazer Dispose interno
                cacheManager.CacheStep(stepCache);
                
                if (t % 25 == 0)
                {
                    long currentMemory = GetMemoryMB();
                    Console.WriteLine($"  Timestep {t}: RAM = {currentMemory}MB (delta: {currentMemory - memoryBefore:+0;-0}MB)");
                }
            }

            long memoryAfterCache = GetMemoryMB();
            long deltaCache = memoryAfterCache - memoryBefore;
            
            Console.WriteLine($"\nRAM após 100 timesteps cacheados: {memoryAfterCache}MB (delta: {deltaCache:+0;-0}MB)");
            
            // ✅ VALIDAÇÃO: Delta deve ser próximo de zero (< 50MB)
            if (deltaCache > 50)
            {
                Console.ForegroundColor = ConsoleColor.Red;
                Console.WriteLine($"❌ FALHA: RAM cresceu {deltaCache}MB! Cache não está sendo liberado!");
                Console.ResetColor();
            }
            else
            {
                Console.ForegroundColor = ConsoleColor.Green;
                Console.WriteLine($"✅ SUCESSO: Delta de {deltaCache}MB está dentro do esperado (<50MB)");
                Console.ResetColor();
            }

            // Testa recuperação
            Console.WriteLine("\nTestando recuperação de tensores...");
            for (int t = 0; t < 10; t++)
            {
                using var tensor = cacheManager.RetrieveTensor(t, DiskOnlyCacheManager.TensorNames.HiddenNext);
                // Tensor usado e liberado automaticamente pelo using
            }
            
            long memoryAfterRetrieve = GetMemoryMB();
            Console.WriteLine($"RAM após 10 recuperações: {memoryAfterRetrieve}MB");
            
            // Reset e validação
            cacheManager.Reset();
            GC.Collect();
            GC.WaitForPendingFinalizers();
            
            long memoryFinal = GetMemoryMB();
            Console.WriteLine($"RAM após Reset: {memoryFinal}MB");
            
            Console.WriteLine("\n✅ Teste 1 concluído.");
        }
        catch (Exception ex)
        {
            Console.ForegroundColor = ConsoleColor.Red;
            Console.WriteLine($"❌ ERRO no Teste 1: {ex.Message}");
            Console.WriteLine($"Stack: {ex.StackTrace}");
            Console.ResetColor();
        }
    }

    /// <summary>
    /// Teste 2: Valida TensorPoolStrict.
    /// </summary>
    private void TestTensorPoolStrict()
    {
        Console.WriteLine("\n[Teste 2] TensorPoolStrict - Limites e Reciclagem");
        Console.WriteLine(new string('-', 70));

        try
        {
            using var pool = new TensorPoolStrict(_mathEngine);
            
            // Teste de reciclagem
            Console.WriteLine("Testando reciclagem de tensores...");
            var shape = new[] { 1, 256 };
            
            for (int cycle = 0; cycle < 3; cycle++)
            {
                Console.WriteLine($"\nCiclo {cycle + 1}: Alugando 100 tensores...");
                var tensors = new IMathTensor[100];
                
                for (int i = 0; i < 100; i++)
                {
                    tensors[i] = pool.Rent(shape);
                }
                
                Console.WriteLine($"  Devolvendo 100 tensores...");
                for (int i = 0; i < 100; i++)
                {
                    pool.Return(tensors[i]);
                }
                
                long currentMemory = GetMemoryMB();
                Console.WriteLine($"  RAM: {currentMemory}MB");
            }
            
            // Imprime estatísticas
            Console.WriteLine("\nEstatísticas finais:");
            pool.PrintDetailedStats();
            
            Console.WriteLine("\n✅ Teste 2 concluído.");
        }
        catch (OutOfMemoryException ex)
        {
            Console.ForegroundColor = ConsoleColor.Red;
            Console.WriteLine($"❌ LIMITE EXCEDIDO: {ex.Message}");
            Console.ResetColor();
        }
        catch (Exception ex)
        {
            Console.ForegroundColor = ConsoleColor.Red;
            Console.WriteLine($"❌ ERRO no Teste 2: {ex.Message}");
            Console.ResetColor();
        }
    }

    /// <summary>
    /// Teste 3: Simula ciclo Forward + Backward completo.
    /// </summary>
    private void TestForwardBackwardCycle()
    {
        Console.WriteLine("\n[Teste 3] Ciclo Forward + Backward - Estabilidade de Memória");
        Console.WriteLine(new string('-', 70));

        long memoryStart = GetMemoryMB();
        Console.WriteLine($"RAM inicial: {memoryStart}MB");

        try
        {
            // Simula 10 batches de treinamento
            Console.WriteLine("\nSimulando 10 batches de treinamento...");
            
            for (int batch = 0; batch < 10; batch++)
            {
                // Simula forward pass
                using var pool = new TensorPoolStrict(_mathEngine);
                using var cache = new DiskOnlyCacheManager(_mathEngine, 128, 256);
                
                // Forward: 20 timesteps
                for (int t = 0; t < 20; t++)
                {
                    var stepCache = new LstmStepCache
                    {
                        Input = pool.Rent(new[] { 1, 128 }),
                        HiddenPrev = pool.Rent(new[] { 1, 256 }),
                        CellPrev = pool.Rent(new[] { 1, 256 }),
                        ForgetGate = pool.Rent(new[] { 1, 256 }),
                        InputGate = pool.Rent(new[] { 1, 256 }),
                        CellCandidate = pool.Rent(new[] { 1, 256 }),
                        OutputGate = pool.Rent(new[] { 1, 256 }),
                        CellNext = pool.Rent(new[] { 1, 256 }),
                        TanhCellNext = pool.Rent(new[] { 1, 256 }),
                        HiddenNext = pool.Rent(new[] { 1, 256 })
                    };
                    
                    cache.CacheStep(stepCache);
                    // stepCache já foi liberado por CacheStep()
                }
                
                // Backward: recupera timesteps do disco
                for (int t = 19; t >= 0; t--)
                {
                    using var tensor1 = cache.RetrieveTensor(t, DiskOnlyCacheManager.TensorNames.HiddenNext);
                    using var tensor2 = cache.RetrieveTensor(t, DiskOnlyCacheManager.TensorNames.CellNext);
                    // Libera automaticamente ao sair do using
                }
                
                cache.Reset();
                pool.Trim();
                
                if (batch % 3 == 0)
                {
                    GC.Collect(1, GCCollectionMode.Optimized);
                }
                
                long currentMemory = GetMemoryMB();
                Console.WriteLine($"  Batch {batch + 1}: RAM = {currentMemory}MB (delta: {currentMemory - memoryStart:+0;-0}MB)");
            }
            
            GC.Collect(2, GCCollectionMode.Forced, true, true);
            GC.WaitForPendingFinalizers();
            
            long memoryEnd = GetMemoryMB();
            long totalDelta = memoryEnd - memoryStart;
            
            Console.WriteLine($"\nRAM final: {memoryEnd}MB");
            Console.WriteLine($"Delta total: {totalDelta:+0;-0}MB");
            
            // ✅ VALIDAÇÃO: Delta total deve ser < 100MB
            if (Math.Abs(totalDelta) > 100)
            {
                Console.ForegroundColor = ConsoleColor.Red;
                Console.WriteLine($"❌ FALHA: RAM variou {totalDelta}MB! Possível vazamento!");
                Console.ResetColor();
            }
            else
            {
                Console.ForegroundColor = ConsoleColor.Green;
                Console.WriteLine($"✅ SUCESSO: Memória estável (variação: {totalDelta}MB)");
                Console.ResetColor();
            }
            
            Console.WriteLine("\n✅ Teste 3 concluído.");
        }
        catch (Exception ex)
        {
            Console.ForegroundColor = ConsoleColor.Red;
            Console.WriteLine($"❌ ERRO no Teste 3: {ex.Message}");
            Console.ResetColor();
        }
    }

    /// <summary>
    /// Teste 4: Estresse de memória (1000 timesteps).
    /// </summary>
    private void TestMemoryStability()
    {
        Console.WriteLine("\n[Teste 4] Teste de Estresse - 1000 Timesteps");
        Console.WriteLine(new string('-', 70));

        long memoryStart = GetMemoryMB();
        Console.WriteLine($"RAM inicial: {memoryStart}MB");
        
        var memoryReadings = new System.Collections.Generic.List<long>();

        try
        {
            using var cache = new DiskOnlyCacheManager(_mathEngine, 128, 256);
            
            Console.WriteLine("Cacheando 1000 timesteps...");
            
            for (int t = 0; t < 1000; t++)
            {
                var stepCache = new LstmStepCache
                {
                    Input = _mathEngine.CreateTensor(new[] { 1, 128 }),
                    HiddenPrev = _mathEngine.CreateTensor(new[] { 1, 256 }),
                    CellPrev = _mathEngine.CreateTensor(new[] { 1, 256 }),
                    ForgetGate = _mathEngine.CreateTensor(new[] { 1, 256 }),
                    InputGate = _mathEngine.CreateTensor(new[] { 1, 256 }),
                    CellCandidate = _mathEngine.CreateTensor(new[] { 1, 256 }),
                    OutputGate = _mathEngine.CreateTensor(new[] { 1, 256 }),
                    CellNext = _mathEngine.CreateTensor(new[] { 1, 256 }),
                    TanhCellNext = _mathEngine.CreateTensor(new[] { 1, 256 }),
                    HiddenNext = _mathEngine.CreateTensor(new[] { 1, 256 })
                };
                
                cache.CacheStep(stepCache);
                
                if (t % 100 == 0)
                {
                    long currentMemory = GetMemoryMB();
                    memoryReadings.Add(currentMemory);
                    Console.WriteLine($"  Timestep {t}: RAM = {currentMemory}MB");
                }
            }
            
            Console.WriteLine("\nAnálise de estabilidade:");
            long minMemory = memoryReadings.Min();
            long maxMemory = memoryReadings.Max();
            long avgMemory = (long)memoryReadings.Average();
            long range = maxMemory - minMemory;
            
            Console.WriteLine($"  Mínimo: {minMemory}MB");
            Console.WriteLine($"  Máximo: {maxMemory}MB");
            Console.WriteLine($"  Média:  {avgMemory}MB");
            Console.WriteLine($"  Range:  {range}MB");
            
            // ✅ VALIDAÇÃO: Range deve ser < 200MB (tolerância para variações do OS)
            if (range > 200)
            {
                Console.ForegroundColor = ConsoleColor.Red;
                Console.WriteLine($"❌ FALHA: Range de {range}MB indica crescimento!");
                Console.ResetColor();
            }
            else
            {
                Console.ForegroundColor = ConsoleColor.Green;
                Console.WriteLine($"✅ SUCESSO: Memória estável (range: {range}MB)");
                Console.ResetColor();
            }
            
            Console.WriteLine("\n✅ Teste 4 concluído.");
        }
        catch (Exception ex)
        {
            Console.ForegroundColor = ConsoleColor.Red;
            Console.WriteLine($"❌ ERRO no Teste 4: {ex.Message}");
            Console.ResetColor();
        }
    }

    private long GetMemoryMB()
    {
        _process.Refresh();
        return _process.WorkingSet64 / (1024 * 1024);
    }

    /// <summary>
    /// Executa teste rápido (apenas validações básicas).
    /// </summary>
    public void RunQuickTest()
    {
        Console.WriteLine("\n[TESTE RÁPIDO]");
        Console.WriteLine(new string('-', 70));

        long memBefore = GetMemoryMB();
        
        try
        {
            // Teste básico de cache
            using var cache = new DiskOnlyCacheManager(_mathEngine, 128, 256);
            
            for (int t = 0; t < 50; t++)
            {
                var step = new LstmStepCache
                {
                    Input = _mathEngine.CreateTensor(new[] { 1, 128 }),
                    HiddenNext = _mathEngine.CreateTensor(new[] { 1, 256 })
                };
                cache.CacheStep(step);
            }
            
            long memAfter = GetMemoryMB();
            long delta = memAfter - memBefore;
            
            Console.WriteLine($"RAM: {memBefore}MB → {memAfter}MB (delta: {delta:+0;-0}MB)");
            
            if (delta > 30)
            {
                Console.ForegroundColor = ConsoleColor.Yellow;
                Console.WriteLine($"⚠️  Delta de {delta}MB parece alto. Executar diagnóstico completo.");
                Console.ResetColor();
            }
            else
            {
                Console.ForegroundColor = ConsoleColor.Green;
                Console.WriteLine("✅ Teste rápido OK");
                Console.ResetColor();
            }
        }
        catch (Exception ex)
        {
            Console.ForegroundColor = ConsoleColor.Red;
            Console.WriteLine($"❌ ERRO: {ex.Message}");
            Console.ResetColor();
        }
    }
}

/// <summary>
/// Ponto de entrada para execução standalone da ferramenta.
/// </summary>
public class DiagnosticProgram
{
    public static void Main(string[] args)
    {
        Console.WriteLine("FERRAMENTA DE DIAGNÓSTICO - GALILEU NODE");
        Console.WriteLine("Versão: 1.0");
        Console.WriteLine($"Data: {DateTime.Now:yyyy-MM-dd HH:mm:ss}\n");

        // Tenta GPU, fallback para CPU
        IMathEngine engine;
        try
        {
            engine = new Galileu.Node.Gpu.GpuMathEngine();
            Console.WriteLine("✅ Usando GpuMathEngine\n");
        }
        catch
        {
            engine = new Galileu.Node.Cpu.CpuMathEngine();
            Console.WriteLine("✅ Usando CpuMathEngine (fallback)\n");
        }

        var diagnostic = new DiagnosticTool(engine);

        // Verifica argumentos
        if (args.Length > 0 && args[0] == "--quick")
        {
            diagnostic.RunQuickTest();
        }
        else
        {
            diagnostic.RunFullDiagnostic();
        }

        Console.WriteLine("\nPressione qualquer tecla para sair...");
        Console.ReadKey();
    }
}