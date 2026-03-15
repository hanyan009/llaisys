#include "add_musa.muh"
#include <musa_runtime.h>
#include <musa_fp16.h>
#include <musa_bf16.h>
#include <cstdio>

namespace llaisys::ops::musa {

__device__ float add_func(float a, float b) { return a + b; }
__device__ __half add_func(__half a, __half b) { return __hadd(a, b); }
__device__ __mt_bfloat16 add_func(__mt_bfloat16 a, __mt_bfloat16 b) {
    return __float2bfloat16(__bfloat162float(a) + __bfloat162float(b));
}

template <typename T>
__global__ void add_kernel(T* c, const T* a, const T* b, int n) {
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += gridDim.x * blockDim.x) {
        c[idx] = add_func(a[idx], b[idx]);
    }
}

void add(tensor_t c, tensor_t a, tensor_t b) {
    int n = c->numel();
    int threads_per_block = 256;
    int num_blocks = (n + threads_per_block - 1) / threads_per_block;
    if (num_blocks > 65535) num_blocks = 65535;

    switch (c->dtype())
    {
    case (LLAISYS_DTYPE_F32):
        add_kernel<float><<<num_blocks, threads_per_block>>>(
            (float*)c->data(), (const float*)a->data(), (const float*)b->data(), n);
        break;
    case (LLAISYS_DTYPE_F16):
        add_kernel<__half><<<num_blocks, threads_per_block>>>(
            (__half*)c->data(), (const __half*)a->data(), (const __half*)b->data(), n);
        break;
    case (LLAISYS_DTYPE_BF16):
        add_kernel<__mt_bfloat16><<<num_blocks, threads_per_block>>>(
            (__mt_bfloat16*)c->data(), (const __mt_bfloat16*)a->data(), (const __mt_bfloat16*)b->data(), n);
        break;
    
    default:
        break;
    }
    
    musaError_t err = musaGetLastError();
    if (err != musaSuccess) {
        fprintf(stderr, "Failed to launch add kernel: %s\n", musaGetErrorString(err));
    }
}

} // namespace llaisys::ops::musa
