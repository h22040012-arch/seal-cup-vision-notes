#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <wincrypt.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include "onnxruntime_c_api.h"

#define EXPORT extern "C" __declspec(dllexport)
constexpr int MONO8=17301505, RGB24=35127316;
#pragma pack(push,8)
struct CupResult {
 int32_t size,ok,ng,state,healthy;
 float confidence;
 double elapsed_ms;
 char reason[96],error[384],weights_sha256[65],runtime[32];
};
#pragma pack(pop)
template<size_t N> void text(char(&dst)[N],const std::string& s){std::memset(dst,0,N);std::memcpy(dst,s.data(),std::min(N-1,s.size()));}
std::string utf8(const std::wstring& s){int n=WideCharToMultiByte(CP_UTF8,0,s.c_str(),int(s.size()),nullptr,0,nullptr,nullptr);std::string out(n,0);WideCharToMultiByte(CP_UTF8,0,s.c_str(),int(s.size()),out.data(),n,nullptr,nullptr);return out;}
std::wstring ini(const std::wstring& path,const wchar_t* section,const wchar_t* key,const wchar_t* fallback=L""){
 wchar_t b[4096]{};GetPrivateProfileStringW(section,key,fallback,b,4096,path.c_str());return b;
}
std::string sha256(const std::wstring& p){
 HCRYPTPROV provider=0;HCRYPTHASH hash=0;
 if(!CryptAcquireContextW(&provider,nullptr,nullptr,PROV_RSA_AES,CRYPT_VERIFYCONTEXT))throw std::runtime_error("Cannot initialize SHA256 provider");
 if(!CryptCreateHash(provider,CALG_SHA_256,0,0,&hash)){CryptReleaseContext(provider,0);throw std::runtime_error("Cannot create SHA256 hash");}
 HANDLE file=CreateFileW(p.c_str(),GENERIC_READ,FILE_SHARE_READ,nullptr,OPEN_EXISTING,FILE_ATTRIBUTE_NORMAL,nullptr);
 if(file==INVALID_HANDLE_VALUE){CryptDestroyHash(hash);CryptReleaseContext(provider,0);throw std::runtime_error("Model file is missing or inaccessible");}
 std::array<BYTE,65536> buf;DWORD n=0;bool good=true;
 for(;;){if(!ReadFile(file,buf.data(),DWORD(buf.size()),&n,nullptr)){good=false;break;}if(!n)break;if(!CryptHashData(hash,buf.data(),n,0)){good=false;break;}}
 BYTE digest[32]{};DWORD count=32;
 if(!CryptGetHashParam(hash,HP_HASHVAL,digest,&count,0))good=false;
 CloseHandle(file);CryptDestroyHash(hash);CryptReleaseContext(provider,0);
 if(!good)throw std::runtime_error("Model SHA256 read failed");
 std::ostringstream s;s<<std::hex<<std::setfill('0');for(auto v:digest)s<<std::setw(2)<<int(v);return s.str();
}
struct Profile {int roi[4];double candidate,ok,min_area;int threads;std::wstring root,model;std::string sha;};
Profile profile(const std::wstring& root,int station){
 Profile p{};p.root=root;std::wstring file=root+L"\\profiles.ini";const wchar_t* sec=station==0?L"High":L"Low";
 std::wstring model=ini(file,sec,L"Model"),digest=ini(file,sec,L"SHA256");
 if(model.empty()||model.find(L"..")!=std::wstring::npos||model.find(L':')!=std::wstring::npos||model[0]==L'\\'||model[0]==L'/')throw std::runtime_error("Missing or unsafe model profile path");
 p.model=root+L"\\"+model;p.sha=utf8(digest);
 std::string roi=utf8(ini(file,sec,L"ROI"));std::replace(roi.begin(),roi.end(),',',' ');std::istringstream rs(roi);
 for(int& x:p.roi)if(!(rs>>x))throw std::runtime_error("Invalid ROI profile");
 if(!(0<=p.roi[0]&&p.roi[0]<p.roi[2]&&p.roi[2]<=4024&&0<=p.roi[1]&&p.roi[1]<p.roi[3]&&p.roi[3]<=3036))throw std::runtime_error("ROI outside original image");
 p.candidate=std::stod(ini(file,sec,L"Candidate"));p.ok=std::stod(ini(file,sec,L"OK"));p.min_area=std::stod(ini(file,sec,L"MinArea"));p.threads=std::stoi(ini(file,sec,L"Threads"));
 if(!std::isfinite(p.candidate)||!std::isfinite(p.ok)||!std::isfinite(p.min_area)||!(0<=p.candidate&&p.candidate<=p.ok&&p.ok<=1&&0<=p.min_area&&p.min_area<=1&&p.threads>=1&&p.threads<=64))throw std::runtime_error("Invalid threshold/thread profile");
 if(station==1 && ini(file,sec,L"ROIInModel")!=L"1")throw std::runtime_error("Low model requires full-frame input; ROI is inside ONNX");
 if(p.sha.size()!=64||sha256(p.model)!=p.sha)throw std::runtime_error("Model SHA256 mismatch");
 return p;
}
struct Session {Profile cfg;OrtSession* ort=nullptr;std::string input,output;std::mutex mutex;std::vector<float> tensor;};
struct State {std::mutex init;const OrtApi* api=nullptr;OrtEnv* env=nullptr;HMODULE module=nullptr;std::wstring runtime_root;Session* stations[2]{};};
State& state(){static State* p=new State;return *p;}
void check(OrtStatus* status){if(status){auto a=state().api;std::string msg=a->GetErrorMessage(status);a->ReleaseStatus(status);throw std::runtime_error(msg);}}
void load_runtime(const std::wstring& root){
 auto& s=state();if(s.api){if(s.runtime_root!=root)throw std::runtime_error("Native root changed; restart VM before switching installation");return;}
 s.module=LoadLibraryExW((root+L"\\bin\\CupOrt1201.dll").c_str(),nullptr,LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR|LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
 if(!s.module)throw std::runtime_error("Cannot load private ONNX runtime; Windows error "+std::to_string(GetLastError()));
 using BaseFn=const OrtApiBase* (ORT_API_CALL*)();auto base=reinterpret_cast<BaseFn>(GetProcAddress(s.module,"OrtGetApiBase"));
 if(!base)throw std::runtime_error("Private runtime missing OrtGetApiBase");
 if(std::string(base()->GetVersionString())!="1.20.1")throw std::runtime_error("Wrong private ONNX Runtime version");
 s.api=base()->GetApi(ORT_API_VERSION);if(!s.api)throw std::runtime_error("ONNX C API 20 is unavailable");
 check(s.api->CreateEnv(ORT_LOGGING_LEVEL_WARNING,"CupVmInProcess",&s.env));s.runtime_root=root;
}
void shape(const OrtTypeInfo* info,const std::vector<int64_t>& expected,bool output=false){
 auto a=state().api;const OrtTensorTypeAndShapeInfo* tensor=nullptr;check(a->CastTypeInfoToTensorInfo(info,&tensor));
 if(!tensor)throw std::runtime_error("Expected tensor model interface");
 ONNXTensorElementDataType type;check(a->GetTensorElementType(tensor,&type));if(type!=ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)throw std::runtime_error("Expected float32 model tensor");
 size_t n;check(a->GetDimensionsCount(tensor,&n));std::vector<int64_t> dims(n);check(a->GetDimensions(tensor,dims.data(),n));
 if(dims.size()!=expected.size())throw std::runtime_error("Wrong tensor rank");
 for(size_t i=0;i<n;i++)if(dims[i]!=expected[i]&&!(output&&i==1&&expected[i]==-1))throw std::runtime_error("Wrong tensor shape for selected station");
}
Session& ensure(const std::wstring& root,int station){
 if(station<0||station>1)throw std::runtime_error("Station must be 0=high or 1=low");
 auto& s=state();std::lock_guard<std::mutex> guard(s.init);
 load_runtime(root);if(s.stations[station])return *s.stations[station];
 auto p=std::make_unique<Session>();p->cfg=profile(root,station);auto a=s.api;
 OrtSessionOptions* options=nullptr;check(a->CreateSessionOptions(&options));
 try{check(a->SetIntraOpNumThreads(options,p->cfg.threads));check(a->SetInterOpNumThreads(options,1));check(a->SetSessionExecutionMode(options,ORT_SEQUENTIAL));check(a->SetSessionGraphOptimizationLevel(options,ORT_ENABLE_ALL));check(a->CreateSession(s.env,p->cfg.model.c_str(),options,&p->ort));a->ReleaseSessionOptions(options);options=nullptr;
  size_t inputs,outputs;check(a->SessionGetInputCount(p->ort,&inputs));check(a->SessionGetOutputCount(p->ort,&outputs));if(inputs!=1||outputs!=1)throw std::runtime_error("Expected one model input/output");
  OrtTypeInfo* info=nullptr;check(a->SessionGetInputTypeInfo(p->ort,0,&info));try{shape(info,station==0?std::vector<int64_t>{1,3,640,640}:std::vector<int64_t>{1,3,3036,4024});}catch(...){a->ReleaseTypeInfo(info);throw;}a->ReleaseTypeInfo(info);
  check(a->SessionGetOutputTypeInfo(p->ort,0,&info));try{shape(info,station==0?std::vector<int64_t>{1,300,6}:std::vector<int64_t>{1,-1,6},station==1);}catch(...){a->ReleaseTypeInfo(info);throw;}a->ReleaseTypeInfo(info);
  OrtAllocator* alloc=nullptr;check(a->GetAllocatorWithDefaultOptions(&alloc));char* name=nullptr;
  check(a->SessionGetInputName(p->ort,0,alloc,&name));p->input=name;alloc->Free(alloc,name);
  check(a->SessionGetOutputName(p->ort,0,alloc,&name));p->output=name;alloc->Free(alloc,name);
  OrtModelMetadata* meta=nullptr;check(a->SessionGetModelMetadata(p->ort,&meta));char* names=nullptr;
  check(a->ModelMetadataLookupCustomMetadataMap(meta,alloc,"names",&names));std::string classes=names?names:"";if(names)alloc->Free(alloc,names);a->ReleaseModelMetadata(meta);
  classes.erase(std::remove_if(classes.begin(),classes.end(),[](char c){return c==' '||c=='\r'||c=='\n'||c=='\t';}),classes.end());
  if(classes!=(station==0?"{0:'hp_front',1:'hp_back'}":"{0:'lp_front',1:'lp_back'}"))throw std::runtime_error("Class metadata does not match selected station");
 }catch(...){if(options)a->ReleaseSessionOptions(options);if(p->ort)a->ReleaseSession(p->ort);throw;}
 s.stations[station]=p.release();return *s.stations[station];
}
void validate_frame(const uint8_t* data,int64_t bytes,int width,int height,int format){
 if(!data||width!=4024||height!=3036)throw std::runtime_error("Expected current original 4024x3036 image");
 if(format!=MONO8&&format!=RGB24)throw std::runtime_error("Only MONO8 and RGB24 are supported");
 if(bytes!=int64_t(width)*height*(format==MONO8?1:3))throw std::runtime_error("Invalid image buffer length or row stride");
}
// Fixed point, half-pixel bilinear interpolation matching OpenCV 4.10 INTER_LINEAR uint8.
// Coefficient precision and the two-stage vertical truncation are validated against the original cv2 tensor.
void high_tensor(Session& s,const uint8_t* image,int channels,std::vector<float>& tensor){
 const int* roi=s.cfg.roi;int width=roi[2]-roi[0],height=roi[3]-roi[1];
 double scale=std::min(640.0/width,640.0/height);int nw=int(std::nearbyint(width*scale)),nh=int(std::nearbyint(height*scale));
 int px=int(std::nearbyint((640-nw)/2.0-.1)),py=int(std::nearbyint((640-nh)/2.0-.1));
 tensor.assign(3*640*640,114.0f/255.0f);std::vector<int> xs(nw),a0(nw),a1(nw);
 for(int x=0;x<nw;x++){float f=float((x+.5)*(double(width)/nw)-.5);int k=int(std::floor(f));f-=k;if(k<0){k=0;f=0;}if(k>=width-1){k=width-1;f=0;}xs[x]=k;a0[x]=int(std::nearbyint((1.f-f)*2048.f));a1[x]=int(std::nearbyint(f*2048.f));}
 for(int y=0;y<nh;y++){
  float f=float((y+.5)*(double(height)/nh)-.5);int sy=int(std::floor(f));f-=sy;int b0=int(std::nearbyint((1.f-f)*2048.f)),b1=int(std::nearbyint(f*2048.f));
  int y0=std::clamp(sy,0,height-1)+roi[1],y1=std::clamp(sy+1,0,height-1)+roi[1];
  for(int x=0;x<nw;x++)for(int c=0;c<3;c++){
   int x0=xs[x]+roi[0],x1=std::min(xs[x]+1,width-1)+roi[0],k=channels==1?0:c;
   int h0=int(image[(y0*4024+x0)*channels+k])*a0[x]+int(image[(y0*4024+x1)*channels+k])*a1[x];
   int h1=int(image[(y1*4024+x0)*channels+k])*a0[x]+int(image[(y1*4024+x1)*channels+k])*a1[x];
   int v=((((b0*(h0>>4))>>16)+((b1*(h1>>4))>>16)+2)>>2);
   tensor[c*640*640+(y+py)*640+x+px]=float(v)/255.0f;
  }
 }
}
void preprocess(Session& s,int station,const uint8_t* image,int format,std::vector<float>& tensor){
 int channels=format==MONO8?1:3;
 if(station==0){high_tensor(s,image,channels,tensor);return;}
 size_t plane=size_t(4024)*3036;tensor.resize(3*plane);const float scale=float(1.0/255.0);
 for(size_t i=0;i<plane;i++)for(int c=0;c<3;c++)tensor[c*plane+i]=float(image[i*channels+(channels==1?0:c)])*scale;
}
struct Box{double x1,y1,x2,y2,score;int cls;};
void decide_top1(const Profile& cfg,const float* predictions,size_t count,CupResult& result){
  const float* best=nullptr;
  for(size_t i=0;i<count;i+=6){
   const float* q=predictions+i;
   for(int j=0;j<6;j++)if(!std::isfinite(q[j]))throw std::runtime_error("Nonfinite model output");
   if(!best||q[4]>best[4])best=q;
  }
  result.healthy=1;result.confidence=best?best[4]:0;
  text(result.weights_sha256,cfg.sha);text(result.runtime,"1.20.1");
  std::string reason="no_cup";
  if(best&&best[4]>=cfg.candidate){
   if(best[4]>1)throw std::runtime_error("Invalid confidence");
   int cls=int(std::nearbyint(best[5]));
   if(cls<0||cls>1||std::abs(double(best[5])-cls)>1e-5)throw std::runtime_error("Invalid model class output");
   reason=cls==0?"front":"back";
  }
  result.ok=reason=="front";result.ng=1-result.ok;
  result.state=result.ok?1:reason=="back"?2:0;text(result.reason,reason);
}

void inspect(Session& s,int station,const uint8_t* data,int format,CupResult& result){
 std::lock_guard<std::mutex> guard(s.mutex);preprocess(s,station,data,format,s.tensor);auto a=state().api;
 std::vector<int64_t> dims=station==0?std::vector<int64_t>{1,3,640,640}:std::vector<int64_t>{1,3,3036,4024};
 OrtMemoryInfo* memory=nullptr;OrtValue *input=nullptr,*output=nullptr;check(a->CreateCpuMemoryInfo(OrtArenaAllocator,OrtMemTypeDefault,&memory));
 try{
  check(a->CreateTensorWithDataAsOrtValue(memory,s.tensor.data(),s.tensor.size()*sizeof(float),dims.data(),dims.size(),ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,&input));a->ReleaseMemoryInfo(memory);memory=nullptr;
  const char* in=s.input.c_str();const char* out=s.output.c_str();const OrtValue* values[]={input};check(a->Run(s.ort,nullptr,&in,values,1,&out,1,&output));
  OrtTensorTypeAndShapeInfo* info=nullptr;check(a->GetTensorTypeAndShape(output,&info));size_t count=0;check(a->GetTensorShapeElementCount(info,&count));a->ReleaseTensorTypeAndShapeInfo(info);
  if(count==0||count%6||count>6*10000)throw std::runtime_error("Invalid output element count");
  float* predictions=nullptr;check(a->GetTensorMutableData(output,reinterpret_cast<void**>(&predictions)));
  decide_top1(s.cfg,predictions,count,result);
  a->ReleaseValue(output);output=nullptr;a->ReleaseValue(input);input=nullptr;
 }catch(...){if(output)a->ReleaseValue(output);if(input)a->ReleaseValue(input);if(memory)a->ReleaseMemoryInfo(memory);throw;}
}
EXPORT int __cdecl CupAbiVersion(){return 1;}
EXPORT int __cdecl CupResultSize(){return sizeof(CupResult);}
EXPORT int __cdecl CupInitialize(const wchar_t* root,int station,char* error,int capacity){
 try{if(!root)throw std::runtime_error("Missing native root");ensure(root,station);if(error&&capacity>0)error[0]=0;return 0;}
 catch(const std::exception& e){if(error&&capacity>0){std::strncpy(error,e.what(),capacity-1);error[capacity-1]=0;}return 1;}
 catch(...){if(error&&capacity>0){std::strncpy(error,"Unknown native error",capacity-1);error[capacity-1]=0;}return 1;}
}
EXPORT int __cdecl CupInspect(const wchar_t* root,int station,const uint8_t* data,int64_t bytes,int width,int height,int format,CupResult* result){
 if(!result||result->size!=sizeof(CupResult))return 2;
 std::memset(result,0,sizeof(*result));result->size=sizeof(*result);result->ng=1;result->state=4;text(result->reason,"inspection_error");
 auto start=std::chrono::steady_clock::now();int code=0;
 try{validate_frame(data,bytes,width,height,format);if(!root)throw std::runtime_error("Missing native root");auto& session=ensure(root,station);inspect(session,station,data,format,*result);}
 catch(const std::exception& e){result->ok=0;result->ng=1;result->state=4;result->healthy=0;text(result->reason,"native_error");text(result->error,e.what());code=1;}
 catch(...){result->ok=0;result->ng=1;result->state=4;result->healthy=0;text(result->reason,"native_error");text(result->error,"Unknown native error");code=1;}
 result->elapsed_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();return code;
}
EXPORT int __cdecl CupDebugPreprocess(const wchar_t* root,int station,const uint8_t* data,int64_t bytes,int format,float* output,int64_t count){
 try{validate_frame(data,bytes,4024,3036,format);auto& s=ensure(root,station);std::vector<float> tensor;preprocess(s,station,data,format,tensor);if(!output||count!=int64_t(tensor.size()))return 2;std::copy(tensor.begin(),tensor.end(),output);return 0;}catch(...){return 1;}
}
