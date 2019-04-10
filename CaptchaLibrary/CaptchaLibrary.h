#include <iostream>
using namespace std;
extern "C" int __stdcall InitSession();
extern "C" __declspec(dllexport) char *PredictFile(const char *image_path);
extern "C" __declspec(dllexport) char *PredictBase64 (const char *img_base64);
extern "C" __declspec(dllexport) char *PredictBinData(const unsigned char *img_data, const unsigned int& data_size);
extern "C" __declspec(dllexport) int  FreePredictData(char *& pPredictData);
extern "C" void __stdcall FreeSession();
