// --- START OF FILE Brain/Gpu/OpenCLService.cs (FINAL CORRECTED VERSION) ---

using OpenCL.Net;
using System.Linq;
using System.Text;
using Environment = System.Environment;

namespace Galileu.Node.Brain.Gpu;

public class OpenCLService : IDisposable
{
    public bool IsGpuAvailable { get; }
    public Context? Context { get; private set; }
    public Device? Device { get; private set; }
    public CommandQueue? CommandQueue { get; private set; }
    public Dictionary<string, Kernel> Kernels { get; } = new();

    public OpenCLService()
    {
        OpenCL.Net.Program program = default;
        // CORREÇÃO: Usaremos uma flag booleana para controlar a liberação do recurso.
        bool programCreated = false;

        try
        {
            ErrorCode error;

            // 1. Encontrar uma plataforma
            Platform[] platforms = Cl.GetPlatformIDs(out error);
            if (error != ErrorCode.Success || platforms.Length == 0)
            {
                Console.WriteLine("[OpenCL] Nenhuma plataforma OpenCL encontrada.");
                IsGpuAvailable = false;
                return;
            }

            Platform platform = platforms.First();

            // 2. Encontrar um dispositivo de GPU
            Device[] devices = Cl.GetDeviceIDs(platform, DeviceType.Gpu, out error);
            if (error != ErrorCode.Success || devices.Length == 0)
            {
                Console.WriteLine("[OpenCL] Nenhuma GPU compatível com OpenCL encontrada. O modelo usará a CPU.");
                IsGpuAvailable = false;
                return;
            }

            Device = devices.First();

            string deviceName = Cl.GetDeviceInfo(Device.Value, DeviceInfo.Name, out error).ToString();
            Console.WriteLine($"[OpenCL] Dispositivo GPU encontrado: {deviceName}");

            // 3. Criar contexto e fila de comandos
            Context = Cl.CreateContext(null, 1, new[] { Device.Value }, null, IntPtr.Zero, out error);
            if (error != ErrorCode.Success) throw new Exception($"Erro ao criar contexto OpenCL: {error}");

            CommandQueue = Cl.CreateCommandQueue(Context.Value, Device.Value, CommandQueueProperties.None, out error);
            if (error != ErrorCode.Success) throw new Exception($"Erro ao criar fila de comandos: {error}");

            // 4. Carregar e compilar o programa do kernel
            byte[] kernelBytes =
                File.ReadAllBytes(Path.Combine(Environment.CurrentDirectory, "Kernels", "MatrixOperations.cl"));
            string kernelSource = new UTF8Encoding(false).GetString(kernelBytes);

            program = Cl.CreateProgramWithSource(Context.Value, 1, new[] { kernelSource }, null, out error);
            if (error != ErrorCode.Success) throw new Exception($"Erro ao criar programa OpenCL: {error}");

            // CORREÇÃO: Se a linha acima for bem-sucedida, marcamos a flag como true.
            programCreated = true;

            error = Cl.BuildProgram(program, 1, new[] { Device.Value }, "", null, IntPtr.Zero);
            if (error != ErrorCode.Success)
            {
                string buildLog = Cl.GetProgramBuildInfo(program, Device.Value, ProgramBuildInfo.Log, out _).ToString();
                throw new Exception($"Erro de compilação do Kernel OpenCL: {buildLog}");
            }

            // 5. Extrair todos os kernels
            string[] kernelNames =
            {
                "matmul_forward", "elementwise_add_forward", "elementwise_add_broadcast_forward",
                "elementwise_multiply", "sigmoid_forward", "tanh_forward", "exp_forward"
            };

            foreach (var name in kernelNames)
            {
                Kernel kernel = Cl.CreateKernel(program, name, out error);
                if (error != ErrorCode.Success) throw new Exception($"Erro ao criar kernel '{name}': {error}");
                Kernels[name] = kernel;
            }

            IsGpuAvailable = true;
            Console.WriteLine("[OpenCL] Serviço inicializado com sucesso. A inferência será acelerada por GPU.");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[OpenCL] Falha ao inicializar o serviço OpenCL: {ex.Message}. O modelo usará a CPU.");
            IsGpuAvailable = false;
        }
        finally
        {
            // CORREÇÃO: Verificamos a flag booleana. Esta é a maneira mais segura.
            if (programCreated)
            {
                Cl.ReleaseProgram(program);
            }
        }
    }

    public void Dispose()
    {
        foreach (var kernel in Kernels.Values) Cl.ReleaseKernel(kernel);
        Kernels.Clear();
        if (CommandQueue.HasValue) Cl.ReleaseCommandQueue(CommandQueue.Value);
        if (Context.HasValue) Cl.ReleaseContext(Context.Value);
    }
}