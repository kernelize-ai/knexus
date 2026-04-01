#ifndef RT_CUDA_DEVICE_H
#define RT_CUDA_DEVICE_H

#include <cuda_utils.h>

class CudaDevice {

public:

  CUcontext context;
  CUdevice cudaDeviceRef;
  cudaDeviceProp props;

  CudaDevice(int deviceID) {
    cudaGetDeviceProperties(&props, deviceID);
    cuDeviceGet(&cudaDeviceRef, 0);
    #if CUDA_VERSION >= 13000
    CUctxCreateParams params = {0};
    cuCtxCreate(&context, &params, 0, cudaDeviceRef);
    #else
    cuCtxCreate(&context, 0, cudaDeviceRef);
    #endif
  }
  ~CudaDevice() = default;
};

#endif // RT_CUDA_DEVICE_H
