#include "../../../tensor/tensor.hpp"
#include "../../../core/llaisys_core.hpp"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdio>

namespace llaisys::ops::nvidia {

template <typename T>
__global__ void add_kernel(T* c, const T* a, const T* b, int n) {
    // Grid-Stride Loop: 
    // 线程从 idx 开始，每次跨越 gridDim.x * blockDim.x (整个网格的线程数) 处理下一个元素
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += gridDim.x * blockDim.x) {
        c[idx] = a[idx] + b[idx];
    }
}

void add(tensor_t c, tensor_t a, tensor_t b) {
    int n = c->numel();
    int threads_per_block = 256;
    
    // 计算需要的 Block 数量，但设置一个合理的上限以避免创建过多 Block
    // 假设我们不希望超过 65535 个 Block (或者根据 SM 数量动态调整)
    int num_blocks = (n + threads_per_block - 1) / threads_per_block;
    // 限制最大 Block 数，防止 N 很大时超出 Grid 限制
    // 现代 GPU Grid X 维限制通常很大 (2^31-1)，但限制一下通常更稳健
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
        add_kernel<__nv_bfloat16><<<num_blocks, threads_per_block>>>(
            (__nv_bfloat16*)c->data(), (const __nv_bfloat16*)a->data(), (const __nv_bfloat16*)b->data(), n);
        break;
    
    default:
        break;
    }
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to launch add kernel: %s\n", cudaGetErrorString(err));
    }
}

} // namespace llaisys::ops::nvidia
