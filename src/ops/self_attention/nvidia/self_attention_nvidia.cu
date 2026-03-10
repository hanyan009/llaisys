#include "self_attention_nvidia.cuh"
#include "../../../device/nvidia/nvidia_resource.cuh"
#include "../../../tensor/tensor.hpp"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>
#include <iostream>
#include <vector>
#include <cmath>

namespace llaisys::ops::nvidia {

// Helper for data type conversion
template <typename T>
struct CudaDataType;

template <>
struct CudaDataType<float> {
    using type = float;
    static const cudaDataType_t value = CUDA_R_32F;
    static const float one;
    static const float zero;
};
const float CudaDataType<float>::one = 1.0f;
const float CudaDataType<float>::zero = 0.0f;

template <>
struct CudaDataType<__half> {
    using type = __half;
    static const cudaDataType_t value = CUDA_R_16F;
    static const __half one;
    static const __half zero;
};
const __half CudaDataType<__half>::one = __float2half(1.0f);
const __half CudaDataType<__half>::zero = __float2half(0.0f);

template <>
struct CudaDataType<__nv_bfloat16> {
    using type = __nv_bfloat16;
    static const cudaDataType_t value = CUDA_R_16BF;
    static const __nv_bfloat16 one;
    static const __nv_bfloat16 zero;
};
const __nv_bfloat16 CudaDataType<__nv_bfloat16>::one = __float2bfloat16(1.0f);
const __nv_bfloat16 CudaDataType<__nv_bfloat16>::zero = __float2bfloat16(0.0f);


// Softmax Kernel
template <typename T>
__global__ void softmax_kernel(T* x, int rows, int cols) {
    int row = blockIdx.x;
    if (row >= rows) return;

    T* row_ptr = x + row * cols;
    int limit = (cols - rows) + row;

    // Find max
    float max_val = -1e20f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float val;
        if constexpr (std::is_same_v<T, float>) val = row_ptr[i];
        else if constexpr (std::is_same_v<T, __half>) val = __half2float(row_ptr[i]);
        else if constexpr (std::is_same_v<T, __nv_bfloat16>) val = __bfloat162float(row_ptr[i]);
        
        if (i > limit) val = -1e20f;
        
        if (val > max_val) max_val = val;
    }

    // Warp reduction for max
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        float other = __shfl_down_sync(0xffffffff, max_val, offset);
        if (other > max_val) max_val = other;
    }
    // Block reduction (assuming blockDim.x <= 1024 and single block per row)
    // Simplified: Use shared memory
    __shared__ float s_max;
    if (threadIdx.x == 0) s_max = max_val;
    __syncthreads();
    // Only thread 0 writes to shared, but we need all threads to agree on max.
    // Actually, proper reduction is needed.
    // For simplicity, let's assume warp reduction is enough if cols is small? No.
    // Let's implement proper block reduction using shared memory.
    
    // ... skipping complex reduction for now, using atomicMax? No float atomicMax is tricky.
    // Let's rely on a simpler approach: 
    // Just a loop in thread 0? No, slow.
    
    // Standard block reduction
    __shared__ float s_data[32]; // For warping
    
    float val = max_val;
    // We already did warp reduction. 
    // Now threads 0, 32, 64... have partial maxes.
    if (threadIdx.x % 32 == 0) {
        s_data[threadIdx.x / 32] = val;
    }
    __syncthreads();
    
    if (threadIdx.x < 32) {
        // Reduce s_data
        float my_val = (threadIdx.x < (blockDim.x + 31) / 32) ? s_data[threadIdx.x] : -1e20f;
        for (int offset = 16; offset > 0; offset /= 2) {
             float other = __shfl_down_sync(0xffffffff, my_val, offset);
             if (other > my_val) my_val = other;
        }
        if (threadIdx.x == 0) s_max = my_val;
    }
    __syncthreads();
    max_val = s_max;

    // Exp and Sum
    float sum = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float val;
        if constexpr (std::is_same_v<T, float>) val = row_ptr[i];
        else if constexpr (std::is_same_v<T, __half>) val = __half2float(row_ptr[i]);
        else if constexpr (std::is_same_v<T, __nv_bfloat16>) val = __bfloat162float(row_ptr[i]);
        
        if (i > limit) val = -1e20f;
        
        float res = expf(val - max_val);
        sum += res;
        
        // Write back temporarily (or recompute later)
        if constexpr (std::is_same_v<T, float>) row_ptr[i] = res;
        else if constexpr (std::is_same_v<T, __half>) row_ptr[i] = __float2half(res);
        else if constexpr (std::is_same_v<T, __nv_bfloat16>) row_ptr[i] = __float2bfloat16(res);
    }
    
    // Reduce sum
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    if (threadIdx.x % 32 == 0) s_data[threadIdx.x / 32] = sum;
    __syncthreads();
    
    __shared__ float s_sum;
    if (threadIdx.x < 32) {
        float my_val = (threadIdx.x < (blockDim.x + 31) / 32) ? s_data[threadIdx.x] : 0.0f;
        for (int offset = 16; offset > 0; offset /= 2) {
             my_val += __shfl_down_sync(0xffffffff, my_val, offset);
        }
        if (threadIdx.x == 0) s_sum = my_val;
    }
    __syncthreads();
    sum = s_sum;
    
    // Normalize
    float inv_sum = 1.0f / (sum + 1e-6f);
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float val;
        if constexpr (std::is_same_v<T, float>) val = row_ptr[i];
        else if constexpr (std::is_same_v<T, __half>) val = __half2float(row_ptr[i]);
        else if constexpr (std::is_same_v<T, __nv_bfloat16>) val = __bfloat162float(row_ptr[i]);
        
        val *= inv_sum;
        
        if constexpr (std::is_same_v<T, float>) row_ptr[i] = val;
        else if constexpr (std::is_same_v<T, __half>) row_ptr[i] = __float2half(val);
        else if constexpr (std::is_same_v<T, __nv_bfloat16>) row_ptr[i] = __float2bfloat16(val);
    }
}

template <typename T>
void launch_softmax(T* data, int rows, int cols) {
    int blockSize = 256;
    softmax_kernel<<<rows, blockSize>>>(data, rows, cols);
}

template <typename T>
void self_attention_impl(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale, cublasHandle_t handle) {
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

    // Create temporary tensor for scores [seq_len, kv_len]
    // We reuse this for each head.
    auto scores_tensor = Tensor::create({qlen, kvlen}, attn_val->dtype(), attn_val->deviceType(), attn_val->deviceId());
    T* d_scores = (T*)scores_tensor->data();

    T alpha;
    if constexpr (std::is_same_v<T, float>) alpha = scale;
    else if constexpr (std::is_same_v<T, __half>) alpha = __float2half(scale);
    else if constexpr (std::is_same_v<T, __nv_bfloat16>) alpha = __float2bfloat16(scale);

    float alpha_f = scale;
    float beta_f = 0.0f;
    float one_f = 1.0f;
    
    // For half/bf16, we usually use float accumulation
    
    for (size_t h = 0; h < nh; ++h) {
        size_t h_kv = h / (nh / nkvh);
        
        // Pointers to the start of the head column
        // q[0][h] is at d_q + h * hd
        // k[0][h_kv] is at d_k + h_kv * hd
        
        T* cur_q = d_q + h * hd;
        T* cur_k = d_k + h_kv * hd;
        T* cur_v = d_v + h_kv * hd;
        T* cur_out = d_out + h * hd;

        // 1. Scores = Q * K^T
        // S (seq, kv) = Q (seq, hd) * K^T (hd, kv)
        // CuBLAS: S^T = (K^T)^T * Q^T = K * Q^T
        // A=K (kv, hd), B=Q (seq, hd), C=S^T (kv, seq) -> No wait.
        // We want S row-major.
        // gemm args: m=kv, n=seq, k=hd.
        // A=K, lda=nkvh*hd (stride between rows of K).
        // B=Q, ldb=nh*hd.
        // C=S, ldc=kv.
        // TransA=T, TransB=N.
        // Result is S^T (kv x seq) in col-major. Which is S (seq x kv) in row-major.
        
        // Wait, type specific gemm
        if constexpr (std::is_same_v<T, float>) {
            cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                        kvlen, qlen, hd,
                        &alpha_f,
                        (const float*)cur_k, kv_stride_row,
                        (const float*)cur_q, q_stride_row,
                        &beta_f,
                        (float*)d_scores, kvlen);
        } else if constexpr (std::is_same_v<T, __half>) {
            __half alpha_h = __float2half(scale);
            __half beta_h = __float2half(0.0f);
            cublasHgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                        kvlen, qlen, hd,
                        &alpha_h,
                        (const __half*)cur_k, kv_stride_row,
                        (const __half*)cur_q, q_stride_row,
                        &beta_h,
                        (__half*)d_scores, kvlen);
        } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
            // cublasGemmEx required for bf16?
             float alpha_f = scale;
             float beta_f = 0.0f;
             cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                        kvlen, qlen, hd,
                        &alpha_f,
                        cur_k, CUDA_R_16BF, kv_stride_row,
                        cur_q, CUDA_R_16BF, q_stride_row,
                        &beta_f,
                        d_scores, CUDA_R_16BF, kvlen,
                        CUDA_R_32F, CUBLAS_GEMM_DEFAULT);
        }

        // 2. Softmax
        launch_softmax(d_scores, qlen, kvlen);

        // 3. Out = Scores * V
        // Out (seq, hd) = Scores (seq, kv) * V (kv, hd)
        // CuBLAS: Out^T = V^T * Scores^T
        // m=hd, n=seq, k=kv.
        // A=V, lda=nkvh*hd.
        // B=Scores, ldb=kv.
        // C=Out, ldc=nh*hd.
        // TransA=N, TransB=N.
        
        if constexpr (std::is_same_v<T, float>) {
            cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                        hd, qlen, kvlen,
                        &one_f,
                        (const float*)cur_v, kv_stride_row,
                        (const float*)d_scores, kvlen,
                        &beta_f,
                        (float*)cur_out, out_stride_row);
        } else if constexpr (std::is_same_v<T, __half>) {
            __half one_h = __float2half(1.0f);
            __half beta_h = __float2half(0.0f);
            cublasHgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                        hd, qlen, kvlen,
                        &one_h,
                        (const __half*)cur_v, kv_stride_row,
                        (const __half*)d_scores, kvlen,
                        &beta_h,
                        (__half*)cur_out, out_stride_row);
        } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
            float one_f = 1.0f;
            float beta_f = 0.0f;
            cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                        hd, qlen, kvlen,
                        &one_f,
                        cur_v, CUDA_R_16BF, kv_stride_row,
                        d_scores, CUDA_R_16BF, kvlen,
                        &beta_f,
                        cur_out, CUDA_R_16BF, out_stride_row,
                        CUDA_R_32F, CUBLAS_GEMM_DEFAULT);
        }
    }
}


void self_attention(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale) {
    int device_id = attn_val->deviceId();
    auto resource = llaisys::device::nvidia::get_resource(device_id);
    ASSERT(resource != nullptr, "NVIDIA resource not found.");
    cublasHandle_t handle = resource->cublas_handle();

    auto dtype = attn_val->dtype();
    if (dtype == LLAISYS_DTYPE_F32) {
        self_attention_impl<float>(attn_val, q, k, v, scale, handle);
    } else if (dtype == LLAISYS_DTYPE_F16) {
        self_attention_impl<__half>(attn_val, q, k, v, scale, handle);
    } else if (dtype == LLAISYS_DTYPE_BF16) {
        self_attention_impl<__nv_bfloat16>(attn_val, q, k, v, scale, handle);
    } else {
        EXCEPTION_DATATYPE_MISMATCH;
    }
}

} // namespace llaisys::ops::nvidia
