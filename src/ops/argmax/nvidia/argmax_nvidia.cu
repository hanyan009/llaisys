#include "argmax_nvidia.cuh"
#include "../../../core/llaisys_core.hpp"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cstddef>
#include <cstdint>
#include <cmath>

namespace llaisys::ops::nvidia {

// --- Helper functions ---
__device__ __forceinline__ float to_float(float val) { return val; }

__device__ __forceinline__ float to_float(__half val) {
    return __half2float(val);
}

__device__ __forceinline__ float to_float(__nv_bfloat16 val) {
    return __bfloat162float(val);
}

// --- CUDA Kernel ---
template <typename T>
__global__ void argmax_kernel(int64_t *max_idx, T *max_val, const T *vals, size_t numel) {
    __shared__ float s_vals[1024];
    __shared__ int64_t s_idxs[1024];

    size_t tid = threadIdx.x;
    
    float local_max = -INFINITY; 
    int64_t local_idx = -1;

    // Grid-Stride Loop
    for (size_t i = tid; i < numel; i += blockDim.x) {
        float val = to_float(vals[i]);
        if (val > local_max || local_idx == -1) {
            local_max = val;
            local_idx = i;
        }
    }

    s_vals[tid] = local_max;
    s_idxs[tid] = local_idx;
    __syncthreads();

    // Reduction
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (s_vals[tid + s] > s_vals[tid]) {
                s_vals[tid] = s_vals[tid + s];
                s_idxs[tid] = s_idxs[tid + s];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        *max_idx = s_idxs[0];
        if (s_idxs[0] != -1) {
            *max_val = vals[s_idxs[0]];
        }
    }
}

// --- Host Implementation ---
template <typename T>
void argmax_impl(std::byte *max_idx, std::byte *max_val, const std::byte *vals, size_t numel) {
    dim3 block(1024);
    dim3 grid(1);

    auto d_max_idx = reinterpret_cast<int64_t*>(max_idx);
    auto d_max_val = reinterpret_cast<T*>(max_val);
    auto d_vals = reinterpret_cast<const T*>(vals);

    argmax_kernel<T><<<grid, block>>>(d_max_idx, d_max_val, d_vals, numel);
}

void argmax(std::byte *max_idx,
            std::byte *max_val,
            const std::byte *vals,
            llaisysDataType_t val_type,
            llaisysDataType_t idx_type,
            size_t numel) {
    
    if (idx_type != LLAISYS_DTYPE_I64) {
        fprintf(stderr, "Argmax: only INT64 index type is supported for now.\n");
        return;
    }
    
    if (numel == 0) return;

    switch (val_type) {
    case LLAISYS_DTYPE_F32:
        argmax_impl<float>(max_idx, max_val, vals, numel);
        break;
    case LLAISYS_DTYPE_F16:
        argmax_impl<__half>(max_idx, max_val, vals, numel);
        break;
    case LLAISYS_DTYPE_BF16:
        argmax_impl<__nv_bfloat16>(max_idx, max_val, vals, numel);
        break;
    default:
        fprintf(stderr, "Argmax: unsupported data type %d\n", val_type);
        break;
    }
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to launch argmax kernel: %s\n", cudaGetErrorString(err));
    }
}

} // namespace llaisys::ops::nvidia
