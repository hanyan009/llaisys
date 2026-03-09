#include "../../../tensor/tensor.hpp"
#include "../../../core/llaisys_core.hpp"
#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace llaisys::ops::nvidia {

// --- 辅助函数：将不同类型转换为 float 以便比较 ---
__device__ __forceinline__ float to_float(float val) { return val; }

__device__ __forceinline__ float to_float(llaisys::fp16_t val) {
    // 将自定义 fp16 结构体转为 CUDA 的 __half 类型，再转为 float
    return __half2float(*reinterpret_cast<__half*>(&val));
}

__device__ __forceinline__ float to_float(llaisys::bf16_t val) {
    // BFloat16 转 Float 的简单位操作：左移 16 位
    uint32_t bits = static_cast<uint32_t>(val._v) << 16;
    return *reinterpret_cast<float*>(&bits);
}

// --- CUDA Kernel 实现 ---
// 使用最简单的单 Block 规约算法
template <typename T>
__global__ void argmax_kernel(int64_t *max_idx, T *max_val, const T *vals, size_t numel) {
    // 申请共享内存，用于在 Block 内部交换数据
    // 假设 Block 大小为 1024，需要存储 1024 个值和索引
    __shared__ float s_vals[1024];
    __shared__ int64_t s_idxs[1024];

    size_t tid = threadIdx.x;
    
    // 1. 每个线程计算自己负责的数据片段的最大值
    // 初始化局部最大值为负无穷
    float local_max = -INFINITY; 
    int64_t local_idx = -1;

    // Grid-Stride Loop: 即使数据量很大，也能通过循环处理完
    // 这里我们只启动 1 个 Block，所以 stride 就是 blockDim.x
    for (size_t i = tid; i < numel; i += blockDim.x) {
        float val = to_float(vals[i]);
        // 如果当前值更大，或者这是第一个有效值，则更新
        if (val > local_max || local_idx == -1) {
            local_max = val;
            local_idx = i;
        }
    }

    // 2. 将每个线程的计算结果写入共享内存
    s_vals[tid] = local_max;
    s_idxs[tid] = local_idx;
    __syncthreads(); // 等待所有线程完成写入

    // 3. 在共享内存中进行树状规约 (Tree Reduction)
    // 每一轮将有效元素减半：1024 -> 512 -> 256 ... -> 1
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            // 比较 tid 和 tid + s 位置的值
            if (s_vals[tid + s] > s_vals[tid]) {
                s_vals[tid] = s_vals[tid + s];
                s_idxs[tid] = s_idxs[tid + s];
            }
        }
        __syncthreads(); // 每一轮比较后都需要同步
    }

    // 4. 最终结果位于共享内存的 0 号位置，由 0 号线程写入全局内存
    if (tid == 0) {
        *max_idx = s_idxs[0];
        // 直接从原始数组读取原始类型的值写入输出，避免精度损失
        if (s_idxs[0] != -1) {
            *max_val = vals[s_idxs[0]];
        }
    }
}

// --- Host 端调用接口 ---
template <typename T>
void argmax_impl(std::byte *max_idx, std::byte *max_val, const std::byte *vals, size_t numel) {
    // 启动 1 个 Block，1024 个线程
    // 这是最简单的实现，虽然对超大数组效率不是极致，但逻辑清晰且正确
    dim3 block(1024);
    dim3 grid(1);

    // 转换指针类型
    auto d_max_idx = reinterpret_cast<int64_t*>(max_idx);
    auto d_max_val = reinterpret_cast<T*>(max_val);
    auto d_vals = reinterpret_cast<const T*>(vals);

    // 调用 Kernel
    argmax_kernel<T><<<grid, block>>>(d_max_idx, d_max_val, d_vals, numel);
}

void argmax(std::byte *max_idx,
            std::byte *max_val,
            const std::byte *vals,
            llaisysDataType_t val_type,
            llaisysDataType_t idx_type,
            size_t numel) {
    ASSERT(numel > 0, "Argmax: numel must be > 0.");

    // 索引类型目前仅支持 i64
    if (idx_type != LLAISYS_DTYPE_I64) {
        EXCEPTION_UNSUPPORTED_DATATYPE(idx_type);
    }

    switch (val_type) {
    case LLAISYS_DTYPE_F32:
        argmax_impl<float>(max_idx, max_val, vals, numel);
        break;
    case LLAISYS_DTYPE_F16:
        argmax_impl<llaisys::fp16_t>(max_idx, max_val, vals, numel);
        break;
    case LLAISYS_DTYPE_BF16:
        argmax_impl<llaisys::bf16_t>(max_idx, max_val, vals, numel);
        break;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(val_type);
    }
}

} // namespace llaisys::ops::nvidia
