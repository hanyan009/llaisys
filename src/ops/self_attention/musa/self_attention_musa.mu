#include "self_attention_musa.muh"
#include "../../../device/musa/musa_resource.muh"
#include "../../../tensor/tensor.hpp"
#include <musa_runtime.h>
#include <musa_fp16.h>
#include <musa_bf16.h>
#include <mublas.h>
#include <cstdio>
#include <type_traits>
#include <cmath>

namespace llaisys::ops::musa {

__device__ inline float to_float(float v) { return v; }
__device__ inline float to_float(__half v) { return __half2float(v); }
__device__ inline float to_float(__mt_bfloat16 v) { return __bfloat162float(v); }

template <typename T>
__global__ void softmax_kernel(T* x, int rows, int cols) {
    int row = blockIdx.x;
    if (row >= rows) return;

    T* row_ptr = x + row * cols;
    int limit = (cols - rows) + row;

    float max_val = -1e20f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float val = to_float(row_ptr[i]);
        if (i > limit) val = -1e20f;
        if (val > max_val) max_val = val;
    }

    // Block reduction for max using shared memory
    extern __shared__ float s_data[];
    
    // Warp reduction
    for (int offset = 16; offset > 0; offset /= 2) {
        float other = __shfl_down_sync(0xffffffff, max_val, offset);
        if (other > max_val) max_val = other;
    }
    
    if (threadIdx.x % 32 == 0) {
        s_data[threadIdx.x / 32] = max_val;
    }
    __syncthreads();
    
    if (threadIdx.x < 32) {
        float my_val = (threadIdx.x < (blockDim.x + 31) / 32) ? s_data[threadIdx.x] : -1e20f;
        for (int offset = 16; offset > 0; offset /= 2) {
             float other = __shfl_down_sync(0xffffffff, my_val, offset);
             if (other > my_val) my_val = other;
        }
        if (threadIdx.x == 0) s_data[0] = my_val;
    }
    __syncthreads();
    max_val = s_data[0];

    // Exp and Sum
    float sum = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float val = to_float(row_ptr[i]);
        if (i > limit) val = -1e20f;
        
        float res = expf(val - max_val);
        sum += res;
        
        if constexpr (std::is_same_v<T, float>) row_ptr[i] = res;
        else if constexpr (std::is_same_v<T, __half>) row_ptr[i] = __float2half(res);
        else if constexpr (std::is_same_v<T, __mt_bfloat16>) row_ptr[i] = __float2bfloat16(res);
    }
    
    // Warp reduction for sum
    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    if (threadIdx.x % 32 == 0) s_data[threadIdx.x / 32] = sum;
    __syncthreads();
    
    if (threadIdx.x < 32) {
        float my_val = (threadIdx.x < (blockDim.x + 31) / 32) ? s_data[threadIdx.x] : 0.0f;
        for (int offset = 16; offset > 0; offset /= 2) {
             my_val += __shfl_down_sync(0xffffffff, my_val, offset);
        }
        if (threadIdx.x == 0) s_data[0] = my_val;
    }
    __syncthreads();
    sum = s_data[0];
    
    float inv_sum = 1.0f / (sum + 1e-6f);
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float val = to_float(row_ptr[i]);
        val *= inv_sum;
        
        if constexpr (std::is_same_v<T, float>) row_ptr[i] = val;
        else if constexpr (std::is_same_v<T, __half>) row_ptr[i] = __float2half(val);
        else if constexpr (std::is_same_v<T, __mt_bfloat16>) row_ptr[i] = __float2bfloat16(val);
    }
}

template <typename T>
void launch_softmax(T* data, int rows, int cols) {
    int blockSize = 256;
    size_t shared_mem = 32 * sizeof(float); // Need enough for warp reduction results
    softmax_kernel<<<rows, blockSize, shared_mem>>>(data, rows, cols);
}

template <typename T>
void self_attention_impl(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale, mublasHandle_t handle) {
    size_t qlen = q->shape()[0];
    size_t nh = q->shape()[1];
    size_t hd = q->shape()[2];
    size_t kvlen = k->shape()[0];
    size_t nkvh = k->shape()[1];

    size_t q_stride_row = nh * hd;
    size_t kv_stride_row = nkvh * hd;
    size_t out_stride_row = nh * hd;

    T* d_q = (T*)q->data();
    T* d_k = (T*)k->data();
    T* d_v = (T*)v->data();
    T* d_out = (T*)attn_val->data();

    auto scores_tensor = Tensor::create({qlen, kvlen}, attn_val->dtype(), attn_val->deviceType(), attn_val->deviceId());
    T* d_scores = (T*)scores_tensor->data();

    float alpha_f = scale;
    float beta_f = 0.0f;
    float one_f = 1.0f;

    for (size_t h = 0; h < nh; ++h) {
        size_t h_kv = h / (nh / nkvh);
        
        T* cur_q = d_q + h * hd;
        T* cur_k = d_k + h_kv * hd;
        T* cur_v = d_v + h_kv * hd;
        T* cur_out = d_out + h * hd;

        if constexpr (std::is_same_v<T, float>) {
            mublasSgemm(handle, MUBLAS_OP_T, MUBLAS_OP_N,
                        kvlen, qlen, hd,
                        &alpha_f,
                        (const float*)cur_k, kv_stride_row,
                        (const float*)cur_q, q_stride_row,
                        &beta_f,
                        (float*)d_scores, kvlen);
        } else if constexpr (std::is_same_v<T, __half>) {
            __half alpha_h = __float2half(scale);
            __half beta_h = __float2half(0.0f);
            mublasHgemm(handle, MUBLAS_OP_T, MUBLAS_OP_N,
                        kvlen, qlen, hd,
                        &alpha_h,
                        (const __half*)cur_k, kv_stride_row,
                        (const __half*)cur_q, q_stride_row,
                        &beta_h,
                        (__half*)d_scores, kvlen);
        } else if constexpr (std::is_same_v<T, __mt_bfloat16>) {
             float alpha_f = scale;
             float beta_f = 0.0f;
             mublasGemmEx(handle, MUBLAS_OP_T, MUBLAS_OP_N,
                        kvlen, qlen, hd,
                        &alpha_f,
                        cur_k, MUSA_R_16BF, kv_stride_row,
                        cur_q, MUSA_R_16BF, q_stride_row,
                        &beta_f,
                        d_scores, MUSA_R_16BF, kvlen,
                        MUBLAS_COMPUTE_32F, MUBLAS_GEMM_DEFAULT);
        }

        launch_softmax(d_scores, qlen, kvlen);

        if constexpr (std::is_same_v<T, float>) {
            mublasSgemm(handle, MUBLAS_OP_N, MUBLAS_OP_N,
                        hd, qlen, kvlen,
                        &one_f,
                        (const float*)cur_v, kv_stride_row,
                        (const float*)d_scores, kvlen,
                        &beta_f,
                        (float*)cur_out, out_stride_row);
        } else if constexpr (std::is_same_v<T, __half>) {
            __half one_h = __float2half(1.0f);
            __half beta_h = __float2half(0.0f);
            mublasHgemm(handle, MUBLAS_OP_N, MUBLAS_OP_N,
                        hd, qlen, kvlen,
                        &one_h,
                        (const __half*)cur_v, kv_stride_row,
                        (const __half*)d_scores, kvlen,
                        &beta_h,
                        (__half*)cur_out, out_stride_row);
        } else if constexpr (std::is_same_v<T, __mt_bfloat16>) {
            float one_f = 1.0f;
            float beta_f = 0.0f;
            mublasGemmEx(handle, MUBLAS_OP_N, MUBLAS_OP_N,
                        hd, qlen, kvlen,
                        &one_f,
                        cur_v, MUSA_R_16BF, kv_stride_row,
                        d_scores, MUSA_R_16BF, kvlen,
                        &beta_f,
                        cur_out, MUSA_R_16BF, out_stride_row,
                        MUBLAS_COMPUTE_32F, MUBLAS_GEMM_DEFAULT);
        }
    }
}

void self_attention(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale) {
    int device_id = attn_val->deviceId();
    auto resource = llaisys::device::musa::get_resource(device_id);
    mublasHandle_t handle = resource->mublas_handle();

    if (attn_val->dtype() == LLAISYS_DTYPE_F32) {
        self_attention_impl<float>(attn_val, q, k, v, scale, handle);
    } else if (attn_val->dtype() == LLAISYS_DTYPE_F16) {
        self_attention_impl<__half>(attn_val, q, k, v, scale, handle);
    } else if (attn_val->dtype() == LLAISYS_DTYPE_BF16) {
        self_attention_impl<__mt_bfloat16>(attn_val, q, k, v, scale, handle);
    }
}

} // namespace llaisys::ops::musa
