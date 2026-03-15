#include "rope_musa.muh"
#include <musa_runtime.h>
#include <musa_fp16.h>
#include <musa_bf16.h>
#include <cstdio>
#include <type_traits>
#include <cmath>

namespace llaisys::ops::musa {

// Helper to convert to float
__device__ inline float to_float(float v) { return v; }
__device__ inline float to_float(__half v) { return __half2float(v); }
__device__ inline float to_float(__mt_bfloat16 v) { return __bfloat162float(v); }

template <typename T>
__global__ void rope_kernel(T* out, const T* in, const int64_t* pos_ids, 
                            int seq_len, int n_heads, int head_dim, float theta) {
    int i = threadIdx.x; 
    int half_dim = head_dim / 2;
    if (i >= half_dim) return;

    int head = blockIdx.y;
    int seq = blockIdx.x;
    
    // Indices
    size_t base_idx = (size_t)seq * n_heads * head_dim + head * head_dim;
    size_t idx_a = base_idx + i;
    size_t idx_b = base_idx + half_dim + i;

    // Load position
    int64_t pos = pos_ids[seq];

    // Compute frequency
    float exp = 2.0f * i / head_dim;
    float freq = (float)pos / powf(theta, exp);

    float sin_v, cos_v;
    sincosf(freq, &sin_v, &cos_v);

    // Load values
    float val_a = to_float(in[idx_a]);
    float val_b = to_float(in[idx_b]);

    // Rotate
    float res_a = val_a * cos_v - val_b * sin_v;
    float res_b = val_b * cos_v + val_a * sin_v; 

    // Store
    if constexpr (std::is_same_v<T, float>) {
        out[idx_a] = res_a;
        out[idx_b] = res_b;
    } else if constexpr (std::is_same_v<T, __half>) {
        out[idx_a] = __float2half(res_a);
        out[idx_b] = __float2half(res_b);
    } else if constexpr (std::is_same_v<T, __mt_bfloat16>) {
        out[idx_a] = __float2bfloat16(res_a);
        out[idx_b] = __float2bfloat16(res_b);
    }
}

void rope(tensor_t out, tensor_t in, tensor_t pos_ids, float theta) {
    size_t seq_len = out->shape()[0];
    size_t n_heads = out->shape()[1];
    size_t head_dim = out->shape()[2];
    
    int half_dim = head_dim / 2;
    
    dim3 grid(seq_len, n_heads);
    dim3 block(half_dim); 
    
    if (out->dtype() == LLAISYS_DTYPE_F32) {
        rope_kernel<float><<<grid, block>>>(
            (float*)out->data(), (const float*)in->data(), (const int64_t*)pos_ids->data(),
            seq_len, n_heads, head_dim, theta);
    } else if (out->dtype() == LLAISYS_DTYPE_F16) {
        rope_kernel<__half><<<grid, block>>>(
            (__half*)out->data(), (const __half*)in->data(), (const int64_t*)pos_ids->data(),
            seq_len, n_heads, head_dim, theta);
    } else if (out->dtype() == LLAISYS_DTYPE_BF16) {
        rope_kernel<__mt_bfloat16><<<grid, block>>>(
            (__mt_bfloat16*)out->data(), (const __mt_bfloat16*)in->data(), (const int64_t*)pos_ids->data(),
            seq_len, n_heads, head_dim, theta);
    }

    musaError_t err = musaGetLastError();
    if (err != musaSuccess) {
        fprintf(stderr, "Failed to launch rope kernel: %s\n", musaGetErrorString(err));
    }
}

} // namespace llaisys::ops::musa
