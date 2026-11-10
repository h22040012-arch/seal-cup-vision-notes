using System;
using System.IO;
using System.Text;
using System.Runtime.InteropServices;
using Script.Methods;

public static class CupRuntime
{
    [StructLayout(LayoutKind.Sequential, Pack=8)]
    public struct NativeResult
    {
        public int Size, OK, NG, State, Healthy;
        public float Score;
        public double ElapsedMs;
        [MarshalAs(UnmanagedType.ByValArray,SizeConst=96)] public byte[] Reason;
        [MarshalAs(UnmanagedType.ByValArray,SizeConst=384)] public byte[] Error;
        [MarshalAs(UnmanagedType.ByValArray,SizeConst=65)] public byte[] Weights;
        [MarshalAs(UnmanagedType.ByValArray,SizeConst=32)] public byte[] Runtime;
    }
    [UnmanagedFunctionPointer(CallingConvention.Cdecl)] delegate int InspectFunction(
        [MarshalAs(UnmanagedType.LPWStr)] string root, int station,
        [In] byte[] buffer, long length, int width, int height, int format, ref NativeResult result);
    [UnmanagedFunctionPointer(CallingConvention.Cdecl)] delegate int IntFunction();
    [DllImport("kernel32.dll",CharSet=CharSet.Unicode,SetLastError=true)]
    static extern IntPtr LoadLibraryEx(string name,IntPtr reserved,uint flags);
    [DllImport("kernel32.dll",CharSet=CharSet.Ansi,SetLastError=true)]
    static extern IntPtr GetProcAddress(IntPtr module,string name);
    static readonly object Gate=new object();
    static IntPtr Module=IntPtr.Zero;
    static InspectFunction Inspect;
    static string Root;
    static T Function<T>(string name) where T:class
    {
        IntPtr address=GetProcAddress(Module,name);
        if(address==IntPtr.Zero)throw new InvalidDataException("Native export missing: "+name);
        return Marshal.GetDelegateForFunctionPointer(address,typeof(T)) as T;
    }
    static void Ensure(string root)
    {
        lock(Gate)
        {
            root=Path.GetFullPath(root).TrimEnd('\\');
            if(Inspect!=null)
            {
                if(!String.Equals(root,Root,StringComparison.OrdinalIgnoreCase))
                    throw new InvalidOperationException("Restart VM before switching native roots");
                return;
            }
            string library=Path.Combine(root,@"bin\CupVmInference20261030.dll");
            Module=LoadLibraryEx(library,IntPtr.Zero,0x00000100|0x00001000);
            if(Module==IntPtr.Zero)throw new InvalidOperationException("Cannot load native inference DLL. Windows error "+Marshal.GetLastWin32Error());
            if(Function<IntFunction>("CupAbiVersion")()!=1 || Function<IntFunction>("CupResultSize")()!=Marshal.SizeOf(typeof(NativeResult)))
                throw new InvalidDataException("Native inference ABI mismatch");
            Inspect=Function<InspectFunction>("CupInspect");Root=root;
        }
    }
    public static string Text(byte[] data)
    {
        if(data==null)return "";
        int n=Array.IndexOf(data,(byte)0);if(n<0)n=data.Length;
        return Encoding.UTF8.GetString(data,0,n);
    }
    public static NativeResult Predict(string root,int station,byte[] buffer,int width,int height,int format)
    {
        Ensure(root);
        NativeResult result=new NativeResult();result.Size=Marshal.SizeOf(typeof(NativeResult));
        result.Reason=new byte[96];result.Error=new byte[384];result.Weights=new byte[65];result.Runtime=new byte[32];
        int code=Inspect(Root,station,buffer,buffer==null?0:buffer.LongLength,width,height,format,ref result);
        if(code!=0 || result.Healthy!=1)
            throw new InvalidOperationException(Text(result.Reason)+": "+Text(result.Error));
        if((result.OK!=0&&result.OK!=1)||result.NG!=1-result.OK||result.State<0||result.State>3||(result.OK==1&&result.State!=1)||Text(result.Runtime)!="1.20.1")
            throw new InvalidDataException("Inconsistent native inference response");
        return result;
    }
}

public partial class UserScript : ScriptMethods, IProcessMethods
{
    const string NativeRoot=@"C:\seal_cup_runtime";
    const int Station=1;
    public void Init() { }
    void Publish(int ok,int state,float score,bool healthy,string reason,string id)
    {
        SetIntValue("OK",ok);SetIntValue("NG",1-ok);
        SetIntValue("State",state);SetIntValue("ServiceHealthy",healthy?1:0);
        SetFloatValue("Score",score);SetStringValue("Result",ok==1?"OK":"NG");
        SetStringValue("Reason",reason);SetStringValue("RequestId",id);
    }
    public bool Process()
    {
        int num=0,readStatus;
        try { readStatus=GetIntValue("num",ref num); }
        catch(Exception ex)
        {
            Publish(0,4,0,false,"counter_read_error: "+ex.Message,Guid.NewGuid().ToString("N"));
            return false;
        }
        if(readStatus!=0||num<0)
        {
            Publish(0,4,0,false,"invalid_slot_counter",Guid.NewGuid().ToString("N"));
            return false;
        }
        // Publish the same original 1/2/3 modulo once, before inference.
        int slot=num%3;if(slot==0)slot=3;SetIntValue("trg1",slot);
        string id=Guid.NewGuid().ToString("N");Publish(0,4,0,false,"inspection_pending",id);
        try
        {
            ImageData frame=new ImageData();
            if(GetImageValue("in0",ref frame)!=0||frame==null)
                throw new InvalidOperationException("Cannot read current image");
            CupRuntime.NativeResult result=CupRuntime.Predict(NativeRoot,Station,frame.Buffer,frame.Width,frame.Heigth,(int)frame.PixelFormat);
            Publish(result.OK,result.State,result.Score,true,CupRuntime.Text(result.Reason),id);
        }
        catch(Exception ex)
        {
            Publish(0,4,0,false,ex.GetType().Name+": "+ex.Message,id);
        }
        return true;
    }
}