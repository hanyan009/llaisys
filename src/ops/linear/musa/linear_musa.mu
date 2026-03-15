#include "linear_musa.muh"
#include "../../../core/llaisys_core.hpp"
#include "../../../device/musa/musa_resource.muh"
#include <musa_runtime.h>
#include <mublas.h>
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
__global__ void add_bias_kernel(T* out, const T* bias, int batch_size, int out_features) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int total = batch_size * out_features;

    for (int i = idx; i < total; i += stride) {
        int col = i % out_features;
        out[i] = add_func(out[i], bias[col]);
    }
}

void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias) {
    int device_id = out->deviceId();
    auto resource = llaisys::device::musa::get_resource(device_id);
    ASSERT(resource != nullptr, "MUSA resource not found.");
    mublasHandle_t handle = resource->mublas_handle();

    size_t batch_size = in->shape()[0];
    size_t in_features = in->shape()[1];
    size_t out_features = weight->shape()[0];

    float alpha_f = 1.0f;
    float beta_f = 0.0f;
    
    __half alpha_h = __float2half(1.0f);
    __half beta_h = __float2half(0.0f);

    if (out->dtype() == LLAISYS_DTYPE_F32) {
        mublasSgemm(handle, MUBLAS_OP_T, MUBLAS_OP_N, 
                    out_features, batch_size, in_features,
                    &alpha_f, 
                    (const float*)weight->data(), in_features,
                    (const float*)in->data(), in_features,
                    &beta_f,
                    (float*)out->data(), out_features);
    } else if (out->dtype() == LLAISYS_DTYPE_F16) {
        mublasHgemm(handle, MUBLAS_OP_T, MUBLAS_OP_N,
                    out_features, batch_size, in_features,
                    &alpha_h,
                    (const __half*)weight->data(), in_features,
                    (const __half*)in->data(), in_features,
                    &beta_h,
                    (__half*)out->data(), out_features);
    } else if (out->dtype() == LLAISYS_DTYPE_BF16) {
        float alpha_bf = 1.0f;
        float beta_bf = 0.0f;
        
        mublasGemmEx(handle, MUBLAS_OP_T, MUBLAS_OP_N,
                     out_features, batch_size, in_features,
                     &alpha_bf,
                     weight->data(), MUSA_R_16BF, in_features,
                     in->data(), MUSA_R_16BF, in_features,
                     &beta_bf,
                     out->data(), MUSA_R_16BF, out_features,
                     MUBLAS_COMPUTE_32F, MUBLAS_GEMM_DEFAULT);
    }

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
             add_bias_kernel<__mt_bfloat16><<<blocks, threads>>>(
                (__mt_bfloat16*)out->data(), (const __mt_bfloat16*)bias->data(), batch_size, out_features);
        }
    }
}

} // namespace llaisys::ops::musa
