#include "rope_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>

template <typename T>
void rope_(T *out, const T *in, const int64_t *pos_ids, size_t seq_len, size_t n_heads, size_t head_dim, float theta) {
    size_t half_dim = head_dim / 2;
    
    for (size_t seq = 0; seq < seq_len; seq++) {
        int64_t position = pos_ids[seq];
        
        for (size_t head = 0; head < n_heads; head++) {
            for (size_t i = 0; i < half_dim; i++) {
                // Calculate frequency: position / theta^(2i / head_dim)
                float freq = static_cast<float>(position) / std::pow(theta, 2.0f * i / head_dim);
                float cos_val = std::cos(freq);
                float sin_val = std::sin(freq);
                
                // Get indices
                size_t base_idx = seq * n_heads * head_dim + head * head_dim;
                size_t a_idx = base_idx + i;
                size_t b_idx = base_idx + half_dim + i;
                
                // Apply rotation
                if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                    float a = llaisys::utils::cast<float>(in[a_idx]);
                    float b = llaisys::utils::cast<float>(in[b_idx]);
                    out[a_idx] = llaisys::utils::cast<T>(a * cos_val - b * sin_val);
                    out[b_idx] = llaisys::utils::cast<T>(b * cos_val + a * sin_val);
                } else {
                    T a = in[a_idx];
                    T b = in[b_idx];
                    out[a_idx] = a * cos_val - b * sin_val;
                    out[b_idx] = b * cos_val + a * sin_val;
                }
            }
        }
    }
}

namespace llaisys::ops::cpu {
void rope(std::byte *out, const std::byte *in, const std::byte *pos_ids, llaisysDataType_t type,
          size_t seq_len, size_t n_heads, size_t head_dim, float theta) {
    const int64_t *pos_ids_ptr = reinterpret_cast<const int64_t *>(pos_ids);
    
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return rope_(reinterpret_cast<float *>(out), reinterpret_cast<const float *>(in),
                     pos_ids_ptr, seq_len, n_heads, head_dim, theta);
    case LLAISYS_DTYPE_BF16:
        return rope_(reinterpret_cast<llaisys::bf16_t *>(out), reinterpret_cast<const llaisys::bf16_t *>(in),
                     pos_ids_ptr, seq_len, n_heads, head_dim, theta);
    case LLAISYS_DTYPE_F16:
        return rope_(reinterpret_cast<llaisys::fp16_t *>(out), reinterpret_cast<const llaisys::fp16_t *>(in),
                     pos_ids_ptr, seq_len, n_heads, head_dim, theta);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
