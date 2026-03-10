#include "embedding_nvidia.cuh"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdio>

namespace llaisys::ops::nvidia {

template <typename T, typename IndexT>
__global__ void embedding_kernel(
    T* out,
    const IndexT* index,
    const T* weight,
    size_t total_elements,
    int hidden_size,
    int vocab_size) {
    
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    for (size_t i = idx; i < total_elements; i += stride) {
        int b = i / hidden_size;
        int h = i % hidden_size;

        IndexT word_idx = index[b];
        if (word_idx >= 0 && word_idx < vocab_size) {
            out[i] = weight[word_idx * hidden_size + h];
        } else {
            // Optional: Set to zero or handle error
            // out[i] = 0; 
        }
    }
}

void embedding(tensor_t out, tensor_t index, tensor_t weight) {
    size_t batch_size = index->numel();
    int hidden_size = weight->shape()[1];
    int vocab_size = weight->shape()[0];
    
    size_t total_elements = batch_size * hidden_size;
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535;

    if (out->dtype() == LLAISYS_DTYPE_F32) {
        if (index->dtype() == LLAISYS_DTYPE_I64) {
             embedding_kernel<float, int64_t><<<blocks, threads>>>(
                (float*)out->data(), (const int64_t*)index->data(), (const float*)weight->data(),
                total_elements, hidden_size, vocab_size);
        } else if (index->dtype() == LLAISYS_DTYPE_I32) {
             embedding_kernel<float, int32_t><<<blocks, threads>>>(
                (float*)out->data(), (const int32_t*)index->data(), (const float*)weight->data(),
                total_elements, hidden_size, vocab_size);
        }
    } else if (out->dtype() == LLAISYS_DTYPE_F16) {
        if (index->dtype() == LLAISYS_DTYPE_I64) {
             embedding_kernel<__half, int64_t><<<blocks, threads>>>(
                (__half*)out->data(), (const int64_t*)index->data(), (const __half*)weight->data(),
                total_elements, hidden_size, vocab_size);
        } else if (index->dtype() == LLAISYS_DTYPE_I32) {
             embedding_kernel<__half, int32_t><<<blocks, threads>>>(
                (__half*)out->data(), (const int32_t*)index->data(), (const __half*)weight->data(),
                total_elements, hidden_size, vocab_size);
        }
    } else if (out->dtype() == LLAISYS_DTYPE_BF16) {
        if (index->dtype() == LLAISYS_DTYPE_I64) {
             embedding_kernel<__nv_bfloat16, int64_t><<<blocks, threads>>>(
                (__nv_bfloat16*)out->data(), (const int64_t*)index->data(), (const __nv_bfloat16*)weight->data(),
                total_elements, hidden_size, vocab_size);
        } else if (index->dtype() == LLAISYS_DTYPE_I32) {
             embedding_kernel<__nv_bfloat16, int32_t><<<blocks, threads>>>(
                (__nv_bfloat16*)out->data(), (const int32_t*)index->data(), (const __nv_bfloat16*)weight->data(),
                total_elements, hidden_size, vocab_size);
        }
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to launch embedding kernel: %s\n", cudaGetErrorString(err));
    }
}

}
