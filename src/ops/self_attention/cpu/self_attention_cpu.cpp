#include "self_attention_cpu.hpp"

#include "../../../utils.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

template <typename T>
void self_attention_(T *attn_val, const T *q, const T *k, const T *v,
                     size_t qlen, size_t kvlen, size_t nh, size_t nkvh, size_t hd, float scale) {
    // Input shapes: q[qlen, nh, hd], k[kvlen, nkvh, hd], v[kvlen, nkvh, hd]
    // Output shape: attn_val[qlen, nh, hd]
    
    // Compute the number of times to repeat k and v heads
    size_t head_repeat = nh / nkvh;
    
    // Allocate temporary buffers for attention scores and weights
    std::vector<float> attn_scores(qlen * kvlen);
    
    // For each query head
    for (size_t h = 0; h < nh; h++) {
        size_t kv_head = h / head_repeat;  // Which k/v head to use
        
        // Compute attention scores: Q @ K^T * scale
        for (size_t i = 0; i < qlen; i++) {
            for (size_t j = 0; j < kvlen; j++) {
                float score = 0.0f;
                
                // Dot product between q[i, h, :] and k[j, kv_head, :]
                for (size_t d = 0; d < hd; d++) {
                    size_t q_idx = i * nh * hd + h * hd + d;
                    size_t k_idx = j * nkvh * hd + kv_head * hd + d;
                    
                    if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                        score += llaisys::utils::cast<float>(q[q_idx]) * llaisys::utils::cast<float>(k[k_idx]);
                    } else {
                        score += static_cast<float>(q[q_idx]) * static_cast<float>(k[k_idx]);
                    }
                }
                
                attn_scores[i * kvlen + j] = score * scale;
            }
        }
        
        // Apply causal mask and softmax for each query position
        for (size_t i = 0; i < qlen; i++) {
            // Apply causal mask: positions beyond (kvlen - qlen + i) should be -inf
            size_t valid_len = kvlen - qlen + i + 1;
            
            // Find max for numerical stability
            float max_score = -std::numeric_limits<float>::infinity();
            for (size_t j = 0; j < valid_len; j++) {
                max_score = std::max(max_score, attn_scores[i * kvlen + j]);
            }
            
            // Apply mask to invalid positions
            for (size_t j = valid_len; j < kvlen; j++) {
                attn_scores[i * kvlen + j] = -std::numeric_limits<float>::infinity();
            }
            
            // Compute exp and sum
            float sum_exp = 0.0f;
            for (size_t j = 0; j < kvlen; j++) {
                float exp_val;
                if (attn_scores[i * kvlen + j] == -std::numeric_limits<float>::infinity()) {
                    exp_val = 0.0f;
                } else {
                    exp_val = std::exp(attn_scores[i * kvlen + j] - max_score);
                }
                attn_scores[i * kvlen + j] = exp_val;
                sum_exp += exp_val;
            }
            
            // Normalize
            for (size_t j = 0; j < kvlen; j++) {
                attn_scores[i * kvlen + j] /= sum_exp;
            }
        }
        
        // Compute output: attn_weights @ V
        for (size_t i = 0; i < qlen; i++) {
            for (size_t d = 0; d < hd; d++) {
                float output = 0.0f;
                
                for (size_t j = 0; j < kvlen; j++) {
                    size_t v_idx = j * nkvh * hd + kv_head * hd + d;
                    
                    if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                        output += attn_scores[i * kvlen + j] * llaisys::utils::cast<float>(v[v_idx]);
                    } else {
                        output += attn_scores[i * kvlen + j] * static_cast<float>(v[v_idx]);
                    }
                }
                
                size_t out_idx = i * nh * hd + h * hd + d;
                if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                    attn_val[out_idx] = llaisys::utils::cast<T>(output);
                } else {
                    attn_val[out_idx] = static_cast<T>(output);
                }
            }
        }
    }
}

namespace llaisys::ops::cpu {
void self_attention(std::byte *attn_val, const std::byte *q, const std::byte *k, const std::byte *v,
                    llaisysDataType_t type, size_t qlen, size_t kvlen, size_t nh, size_t nkvh, size_t hd, float scale) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return self_attention_(reinterpret_cast<float *>(attn_val), reinterpret_cast<const float *>(q),
                               reinterpret_cast<const float *>(k), reinterpret_cast<const float *>(v),
                               qlen, kvlen, nh, nkvh, hd, scale);
    case LLAISYS_DTYPE_BF16:
        return self_attention_(reinterpret_cast<llaisys::bf16_t *>(attn_val), reinterpret_cast<const llaisys::bf16_t *>(q),
                               reinterpret_cast<const llaisys::bf16_t *>(k), reinterpret_cast<const llaisys::bf16_t *>(v),
                               qlen, kvlen, nh, nkvh, hd, scale);
    case LLAISYS_DTYPE_F16:
        return self_attention_(reinterpret_cast<llaisys::fp16_t *>(attn_val), reinterpret_cast<const llaisys::fp16_t *>(q),
                               reinterpret_cast<const llaisys::fp16_t *>(k), reinterpret_cast<const llaisys::fp16_t *>(v),
                               qlen, kvlen, nh, nkvh, hd, scale);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
