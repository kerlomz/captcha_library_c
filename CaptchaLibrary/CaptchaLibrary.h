#include <iostream>
using namespace std;
extern "C" __declspec(dllexport) int InitSession();
extern "C" __declspec(dllexport) char *PredictFile(const char *image_path);
extern "C" __declspec(dllexport) char *PredictBase64(const char * img_base64);
extern "C" __declspec(dllexport) void FreeSession();
