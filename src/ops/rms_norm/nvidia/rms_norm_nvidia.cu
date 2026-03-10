#include "rms_norm_nvidia.cuh"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <type_traits>

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
__global__ void rms_norm_kernel(T* out, const T* in, const T* weight, float eps, int hidden_size) {
    // One block per row
    int row = blockIdx.x;
    int tid = threadIdx.x;
    
    // Shared memory for reduction
    extern __shared__ float sdata[];

    float sum_sq = 0.0f;
    for (int i = tid; i < hidden_size; i += blockDim.x) {
        float val = to_float(in[row * hidden_size + i]);
        sum_sq += val * val;
    }

    // Store in shared memory
    sdata[tid] = sum_sq;
    __syncthreads();

    // Parallel reduction in shared memory
    // Assume blockDim.x is power of 2
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Broadcast result
    float inv_rms = 0.0f;
    if (tid == 0) {
        float mean_sq = sdata[0] / hidden_size;
        inv_rms = rsqrtf(mean_sq + eps);
        sdata[0] = inv_rms;
    }
    __syncthreads();
    inv_rms = sdata[0];

    // Write output
    for (int i = tid; i < hidden_size; i += blockDim.x) {
        float val = to_float(in[row * hidden_size + i]);
        float w = to_float(weight[i]);
        // out = val * inv_rms * w
        // Need to cast back to T
        if constexpr (std::is_same_v<T, float>) {
            out[row * hidden_size + i] = val * inv_rms * w;
        } else if constexpr (std::is_same_v<T, __half>) {
            out[row * hidden_size + i] = __float2half(val * inv_rms * w);
        } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
            out[row * hidden_size + i] = __float2bfloat16(val * inv_rms * w);
        }
    }
}

void rms_norm(tensor_t out, tensor_t in, tensor_t weight, float eps) {
    size_t batch_size = in->shape()[0];
    size_t hidden_size = in->shape()[1];
    
    int threads = 256;
    // Ensure hidden_size fits? The loop handles it.
    // Ensure threads is power of 2 for reduction.
    
    int blocks = batch_size;
    size_t shared_mem = threads * sizeof(float);

    if (out->dtype() == LLAISYS_DTYPE_F32) {
        rms_norm_kernel<float><<<blocks, threads, shared_mem>>>(
            (float*)out->data(), (const float*)in->data(), (const float*)weight->data(), eps, hidden_size);
    } else if (out->dtype() == LLAISYS_DTYPE_F16) {
        rms_norm_kernel<__half><<<blocks, threads, shared_mem>>>(
            (__half*)out->data(), (const __half*)in->data(), (const __half*)weight->data(), eps, hidden_size);
    } else if (out->dtype() == LLAISYS_DTYPE_BF16) {
        rms_norm_kernel<__nv_bfloat16><<<blocks, threads, shared_mem>>>(
            (__nv_bfloat16*)out->data(), (const __nv_bfloat16*)in->data(), (const __nv_bfloat16*)weight->data(), eps, hidden_size);
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to launch rms_norm kernel: %s\n", cudaGetErrorString(err));
    }
}

}
