#include "rearrange_nvidia.cuh"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <iostream>

namespace llaisys::ops::nvidia {

constexpr int MAX_NDIM = 5;

struct TensorMeta {
    size_t shape[MAX_NDIM];
    long long in_strides[MAX_NDIM];
    long long out_strides[MAX_NDIM];
    int ndim;
    size_t numel;
};

template <typename T>
__global__ void rearrange_kernel(T* out, const T* in, TensorMeta meta) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= meta.numel) return;

    size_t temp_idx = idx;
    long long in_offset = 0;
    long long out_offset = 0;

    #pragma unroll
    for (int i = meta.ndim - 1; i >= 0; --i) {
        size_t dim_idx = temp_idx % meta.shape[i];
        temp_idx /= meta.shape[i];
        in_offset += dim_idx * meta.in_strides[i];
        out_offset += dim_idx * meta.out_strides[i];
    }

    out[out_offset] = in[in_offset];
}

void rearrange(tensor_t out, tensor_t in) {
    size_t numel = out->numel();
    if (numel == 0) return;

    int ndim = out->ndim();
    if (ndim > MAX_NDIM) {
        std::cerr << "rearrange: ndim " << ndim << " > MAX_NDIM " << MAX_NDIM << std::endl;
        return;
    }

    TensorMeta meta;
    meta.ndim = ndim;
    meta.numel = numel;
    
    auto shape = out->shape();
    auto out_strides = out->strides();
    auto in_strides = in->strides();

    for(int i=0; i<ndim; ++i) {
        meta.shape[i] = shape[i];
        meta.out_strides[i] = static_cast<long long>(out_strides[i]);
        meta.in_strides[i] = static_cast<long long>(in_strides[i]);
    }

    int blockSize = 256;
    int gridSize = (numel + blockSize - 1) / blockSize;

    auto dtype = out->dtype();
    if (dtype == LLAISYS_DTYPE_F32) {
        rearrange_kernel<float><<<gridSize, blockSize>>>((float*)out->data(), (const float*)in->data(), meta);
    } else if (dtype == LLAISYS_DTYPE_F16) {
        rearrange_kernel<__half><<<gridSize, blockSize>>>((__half*)out->data(), (const __half*)in->data(), meta);
    } else if (dtype == LLAISYS_DTYPE_BF16) {
        rearrange_kernel<__nv_bfloat16><<<gridSize, blockSize>>>((__nv_bfloat16*)out->data(), (const __nv_bfloat16*)in->data(), meta);
    } else if (dtype == LLAISYS_DTYPE_I64) {
        rearrange_kernel<int64_t><<<gridSize, blockSize>>>((int64_t*)out->data(), (const int64_t*)in->data(), meta);
    } else {
         std::cerr << "rearrange: unsupported dtype" << std::endl;
    }
}

} // namespace llaisys::ops::nvidia
