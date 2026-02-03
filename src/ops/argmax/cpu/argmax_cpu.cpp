#include "argmax_cpu.hpp"

#include "../../../utils.hpp"

#include <cstdint>

namespace llaisys::ops::cpu {

namespace {

// 针对不同值类型的通用一维全局 argmax 实现。
// 约定：
// - 输出 max_idx / max_val 都是长度为 1 的标量张量；
// - 索引类型固定为 int64_t（对应 Python 侧的 i64）。
template <typename ValT>
void argmax_impl(std::byte *max_idx,
                 std::byte *max_val,
                 const std::byte *vals,
                 size_t numel) {
    ASSERT(numel > 0, "Argmax: numel must be > 0.");

    using IndexT = int64_t;

    auto *idx_ptr = reinterpret_cast<IndexT *>(max_idx);
    auto *out_val_ptr = reinterpret_cast<ValT *>(max_val);
    auto *in_ptr = reinterpret_cast<const ValT *>(vals);

    IndexT best_idx = 0;
    float best_val_f;

    if constexpr (std::is_same_v<ValT, float>) {
        best_val_f = static_cast<float>(in_ptr[0]);
    } else {
        best_val_f = llaisys::utils::cast<float>(in_ptr[0]);
    }

    for (size_t i = 1; i < numel; ++i) {
        float v_f;
        if constexpr (std::is_same_v<ValT, float>) {
            v_f = static_cast<float>(in_ptr[i]);
        } else {
            v_f = llaisys::utils::cast<float>(in_ptr[i]);
        }
        if (v_f > best_val_f) {
            best_val_f = v_f;
            best_idx = static_cast<IndexT>(i);
        }
    }

    idx_ptr[0] = best_idx;

    if constexpr (std::is_same_v<ValT, float>) {
        out_val_ptr[0] = static_cast<ValT>(best_val_f);
    } else {
        out_val_ptr[0] = llaisys::utils::cast<ValT>(best_val_f);
    }
}

} // namespace

// CPU 上的 argmax 核心逻辑：
// - 目前支持的值类型：F32 / F16 / BF16；
// - 目前支持的索引类型：I64。
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
        return argmax_impl<float>(max_idx, max_val, vals, numel);
    case LLAISYS_DTYPE_F16:
        return argmax_impl<llaisys::fp16_t>(max_idx, max_val, vals, numel);
    case LLAISYS_DTYPE_BF16:
        return argmax_impl<llaisys::bf16_t>(max_idx, max_val, vals, numel);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(val_type);
    }
}

} // namespace llaisys::ops::cpu
