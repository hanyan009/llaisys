#include "rms_norm_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>

template <typename T>
void rms_norm_(T *out, const T *in, const T *weight, size_t batch_size, size_t hidden_size, float eps) {
    // RMS Norm formula: out = in / sqrt(mean(in^2) + eps) * weight
    for (size_t b = 0; b < batch_size; b++) {
        const T *in_row = in + b * hidden_size;
        T *out_row = out + b * hidden_size;

        // Calculate mean of squares
        float sum_squares = 0.0f;
        for (size_t i = 0; i < hidden_size; i++) {
            float val;
            if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                val = llaisys::utils::cast<float>(in_row[i]);
            } else {
                val = static_cast<float>(in_row[i]);
            }
            sum_squares += val * val;
        }
        float mean_squares = sum_squares / hidden_size;
        float rms = std::sqrt(mean_squares + eps);

        // Normalize and scale
        for (size_t i = 0; i < hidden_size; i++) {
            float val;
            float w;
            if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                val = llaisys::utils::cast<float>(in_row[i]);
                w = llaisys::utils::cast<float>(weight[i]);
            } else {
                val = static_cast<float>(in_row[i]);
                w = static_cast<float>(weight[i]);
            }
            float normalized = (val / rms) * w;
            if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                out_row[i] = llaisys::utils::cast<T>(normalized);
            } else {
                out_row[i] = static_cast<T>(normalized);
            }
        }
    }
}

namespace llaisys::ops::cpu {
void rms_norm(std::byte *out, const std::byte *in, const std::byte *weight, llaisysDataType_t type, size_t batch_size,
              size_t hidden_size, float eps) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return rms_norm_(reinterpret_cast<float *>(out), reinterpret_cast<const float *>(in),
                         reinterpret_cast<const float *>(weight), batch_size, hidden_size, eps);
    case LLAISYS_DTYPE_BF16:
        return rms_norm_(reinterpret_cast<llaisys::bf16_t *>(out), reinterpret_cast<const llaisys::bf16_t *>(in),
                         reinterpret_cast<const llaisys::bf16_t *>(weight), batch_size, hidden_size, eps);
    case LLAISYS_DTYPE_F16:
        return rms_norm_(reinterpret_cast<llaisys::fp16_t *>(out), reinterpret_cast<const llaisys::fp16_t *>(in),
                         reinterpret_cast<const llaisys::fp16_t *>(weight), batch_size, hidden_size, eps);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
