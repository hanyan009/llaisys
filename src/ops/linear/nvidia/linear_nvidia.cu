#include "linear_nvidia.cuh"
#include "../../../core/llaisys_core.hpp"
#include "../../../device/nvidia/nvidia_resource.cuh"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

namespace llaisys::ops::nvidia {

template <typename T>
__global__ void add_bias_kernel(T* out, const T* bias, int batch_size, int out_features) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int total = batch_size * out_features;

    for (int i = idx; i < total; i += stride) {
        int col = i % out_features;
        out[i] = out[i] + bias[col];
    }
}

void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias) {
    // Get cublas handle
    int device_id = out->deviceId();
    auto resource = llaisys::device::nvidia::get_resource(device_id);
    ASSERT(resource != nullptr, "NVIDIA resource not found.");
    cublasHandle_t handle = resource->cublas_handle();

    size_t batch_size = in->shape()[0];
    size_t in_features = in->shape()[1];
    size_t out_features = weight->shape()[0];

    // GEMM: out = in * weight^T
    // See thought process for transposition logic.
    // Call: cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, out_features, batch_size, in_features, 
    //                   &alpha, weight, in_features, in, in_features, &beta, out, out_features)
    
    // Alpha and Beta
    float alpha_f = 1.0f;
    float beta_f = 0.0f;
    
    // For half/bf16
    __half alpha_h = __float2half(1.0f);
    __half beta_h = __float2half(0.0f);

    // Run GEMM
    if (out->dtype() == LLAISYS_DTYPE_F32) {
        cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, 
                    out_features, batch_size, in_features,
                    &alpha_f, 
                    (const float*)weight->data(), in_features,
                    (const float*)in->data(), in_features,
                    &beta_f,
                    (float*)out->data(), out_features);
    } else if (out->dtype() == LLAISYS_DTYPE_F16) {
        cublasHgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                    out_features, batch_size, in_features,
                    &alpha_h,
                    (const __half*)weight->data(), in_features,
                    (const __half*)in->data(), in_features,
                    &beta_h,
                    (__half*)out->data(), out_features);
    } else if (out->dtype() == LLAISYS_DTYPE_BF16) {
        // BF16 requires GemmEx
        // CUDA_R_16BF
        float alpha_bf = 1.0f;
        float beta_bf = 0.0f;
        
        cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                     out_features, batch_size, in_features,
                     &alpha_bf,
                     weight->data(), CUDA_R_16BF, in_features,
                     in->data(), CUDA_R_16BF, in_features,
                     &beta_bf,
                     out->data(), CUDA_R_16BF, out_features,
                     CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    }

    // Add bias if present
    if (bias) {
        int total = batch_size * out_features;
        int threads = 256;
        int blocks = (total + threads - 1) / threads;
        if (blocks > 65535) blocks = 65535;

        if (out->dtype() == LLAISYS_DTYPE_F32) {
            add_bias_kernel<float><<<blocks, threads>>>(
                (float*)out->data(), (const float*)bias->data(), batch_size, out_features);
        } else if (out->dtype() == LLAISYS_DTYPE_F16) {
            add_bias_kernel<__half><<<blocks, threads>>>(
                (__half*)out->data(), (const __half*)bias->data(), batch_size, out_features);
        } else if (out->dtype() == LLAISYS_DTYPE_BF16) {
             add_bias_kernel<__nv_bfloat16><<<blocks, threads>>>(
                (__nv_bfloat16*)out->data(), (const __nv_bfloat16*)bias->data(), batch_size, out_features);
        }
    }
}

}
