#include "rms_norm_musa.muh"
#include <musa_runtime.h>
#include <musa_fp16.h>
#include <musa_bf16.h>
#include <cstdio>
#include <type_traits>
#include <cmath>

namespace llaisys::ops::musa {

__device__ inline float to_float(float v) { return v; }
__device__ inline float to_float(__half v) { return __half2float(v); }
__device__ inline float to_float(__mt_bfloat16 v) { return __bfloat162float(v); }

template <typename T>
__global__ void rms_norm_kernel(T* out, const T* in, const T* weight, float eps, int hidden_size) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    
    extern __shared__ float sdata[];

    float sum_sq = 0.0f;
    for (int i = tid; i < hidden_size; i += blockDim.x) {
        float val = to_float(in[row * hidden_size + i]);
        sum_sq += val * val;
    }

    sdata[tid] = sum_sq;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    float inv_rms = 0.0f;
    if (tid == 0) {
        float mean_sq = sdata[0] / hidden_size;
        inv_rms = rsqrtf(mean_sq + eps);
        sdata[0] = inv_rms;
    }
    __syncthreads();
    inv_rms = sdata[0];

    for (int i = tid; i < hidden_size; i += blockDim.x) {
        float val = to_float(in[row * hidden_size + i]);
        float w = to_float(weight[i]);
        float res = val * inv_rms * w;
        
        if constexpr (std::is_same_v<T, float>) {
            out[row * hidden_size + i] = res;
        } else if constexpr (std::is_same_v<T, __half>) {
            out[row * hidden_size + i] = __float2half(res);
        } else if constexpr (std::is_same_v<T, __mt_bfloat16>) {
            out[row * hidden_size + i] = __float2bfloat16(res);
        }
    }
}

void rms_norm(tensor_t out, tensor_t in, tensor_t weight, float eps) {
    size_t batch_size = in->shape()[0];
    size_t hidden_size = in->shape()[1];
    
    int threads = 256;
    int blocks = batch_size;
    size_t shared_mem = threads * sizeof(float);

    if (out->dtype() == LLAISYS_DTYPE_F32) {
        rms_norm_kernel<float><<<blocks, threads, shared_mem>>>(
            (float*)out->data(), (const float*)in->data(), (const float*)weight->data(), eps, hidden_size);
    } else if (out->dtype() == LLAISYS_DTYPE_F16) {
        rms_norm_kernel<__half><<<blocks, threads, shared_mem>>>(
            (__half*)out->data(), (const __half*)in->data(), (const __half*)weight->data(), eps, hidden_size);
    } else if (out->dtype() == LLAISYS_DTYPE_BF16) {
        rms_norm_kernel<__mt_bfloat16><<<blocks, threads, shared_mem>>>(
            (__mt_bfloat16*)out->data(), (const __mt_bfloat16*)in->data(), (const __mt_bfloat16*)weight->data(), eps, hidden_size);
    }

    musaError_t err = musaGetLastError();
    if (err != musaSuccess) {
        fprintf(stderr, "Failed to launch rms_norm kernel: %s\n", musaGetErrorString(err));
    }
}

} // namespace llaisys::ops::musa
