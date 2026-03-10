#include "swiglu_nvidia.cuh"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <type_traits>
#include <cmath>

namespace llaisys::ops::nvidia {

// Helper to convert to float
__device__ inline float to_float(float v) { return v; }
__device__ inline float to_float(__half v) { return __half2float(v); }
__device__ inline float to_float(__nv_bfloat16 v) { return __bfloat162float(v); }

// Helper to convert from float
__device__ inline float from_float_float(float v) { return v; }
__device__ inline __half from_float_half(float v) { return __float2half(v); }
__device__ inline __nv_bfloat16 from_float_bf16(float v) { return __float2bfloat16(v); }

template <typename T>
__global__ void swiglu_kernel(T* out, const T* gate, const T* up, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    
    for (size_t i = idx; i < n; i += stride) {
        float g = to_float(gate[i]);
        float u = to_float(up[i]);
        
        // Swish: x * sigmoid(x) = x / (1 + exp(-x))
        float sigmoid = 1.0f / (1.0f + expf(-g));
        float swish = g * sigmoid;
        
        float res = swish * u;
        
        if constexpr (std::is_same_v<T, float>) {
            out[i] = res;
        } else if constexpr (std::is_same_v<T, __half>) {
            out[i] = __float2half(res);
        } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
            out[i] = __float2bfloat16(res);
        }
    }
}

void swiglu(tensor_t out, tensor_t gate, tensor_t up) {
    size_t n = out->numel();
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535;

    if (out->dtype() == LLAISYS_DTYPE_F32) {
        swiglu_kernel<float><<<blocks, threads>>>(
            (float*)out->data(), (const float*)gate->data(), (const float*)up->data(), n);
    } else if (out->dtype() == LLAISYS_DTYPE_F16) {
        swiglu_kernel<__half><<<blocks, threads>>>(
            (__half*)out->data(), (const __half*)gate->data(), (const __half*)up->data(), n);
    } else if (out->dtype() == LLAISYS_DTYPE_BF16) {
        swiglu_kernel<__nv_bfloat16><<<blocks, threads>>>(
            (__nv_bfloat16*)out->data(), (const __nv_bfloat16*)gate->data(), (const __nv_bfloat16*)up->data(), n);
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to launch swiglu kernel: %s\n", cudaGetErrorString(err));
    }
}

}
