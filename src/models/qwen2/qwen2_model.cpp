#include "qwen2_model.hpp"
#include "../../utils.hpp"
#include <cstring>
#include <cmath>
#include <random>
#include <algorithm>
#include <numeric>

namespace llaisys::models {

static void copy_data(void* dst, const void* src, size_t size, llaisysDeviceType_t device_type, llaisysMemcpyKind_t kind) {
    if (device_type == LLAISYS_DEVICE_CPU) {
        std::memcpy(dst, src, size);
    } else {
        auto& runtime = llaisys::core::context().runtime();
        runtime.api()->memcpy_sync(dst, src, size, kind);
    }
}

Qwen2Model::Qwen2Model(const Qwen2Meta &meta, llaisysDeviceType_t device_type, int device_id)
    : _meta(meta), _device_type(device_type), _device_id(device_id) {
    // Allocate main weights
    _weights.in_embed = Tensor::create({meta.voc, meta.hs}, meta.dtype, device_type, device_id);
    _weights.out_embed = Tensor::create({meta.voc, meta.hs}, meta.dtype, device_type, device_id);
    _weights.out_norm_w = Tensor::create({meta.hs}, meta.dtype, device_type, device_id);
    
    // Allocate layer weights
    _weights.attn_norm_w.resize(meta.nlayer);
    _weights.attn_q_w.resize(meta.nlayer);
    _weights.attn_q_b.resize(meta.nlayer);
    _weights.attn_k_w.resize(meta.nlayer);
    _weights.attn_k_b.resize(meta.nlayer);
    _weights.attn_v_w.resize(meta.nlayer);
    _weights.attn_v_b.resize(meta.nlayer);
    _weights.attn_o_w.resize(meta.nlayer);
    _weights.mlp_norm_w.resize(meta.nlayer);
    _weights.mlp_gate_w.resize(meta.nlayer);
    _weights.mlp_up_w.resize(meta.nlayer);
    _weights.mlp_down_w.resize(meta.nlayer);
    
    for (size_t i = 0; i < meta.nlayer; ++i) {
        _weights.attn_norm_w[i] = Tensor::create({meta.hs}, meta.dtype, device_type, device_id);
        // PyTorch weights are [out_features, in_features]
        _weights.attn_q_w[i] = Tensor::create({meta.hs, meta.hs}, meta.dtype, device_type, device_id);
        _weights.attn_q_b[i] = Tensor::create({meta.hs}, meta.dtype, device_type, device_id);
        _weights.attn_k_w[i] = Tensor::create({meta.nkvh * meta.dh, meta.hs}, meta.dtype, device_type, device_id);
        _weights.attn_k_b[i] = Tensor::create({meta.nkvh * meta.dh}, meta.dtype, device_type, device_id);
        _weights.attn_v_w[i] = Tensor::create({meta.nkvh * meta.dh, meta.hs}, meta.dtype, device_type, device_id);
        _weights.attn_v_b[i] = Tensor::create({meta.nkvh * meta.dh}, meta.dtype, device_type, device_id);
        _weights.attn_o_w[i] = Tensor::create({meta.hs, meta.hs}, meta.dtype, device_type, device_id);
        _weights.mlp_norm_w[i] = Tensor::create({meta.hs}, meta.dtype, device_type, device_id);
        _weights.mlp_gate_w[i] = Tensor::create({meta.di, meta.hs}, meta.dtype, device_type, device_id);
        _weights.mlp_up_w[i] = Tensor::create({meta.di, meta.hs}, meta.dtype, device_type, device_id);
        _weights.mlp_down_w[i] = Tensor::create({meta.hs, meta.di}, meta.dtype, device_type, device_id);
    }
    
    initKVCache();
}

void Qwen2Model::initKVCache() {
    _kv_cache.k_cache.resize(_meta.nlayer);
    _kv_cache.v_cache.resize(_meta.nlayer);
    _kv_cache.seq_len = 0;
    
    for (size_t i = 0; i < _meta.nlayer; ++i) {
        _kv_cache.k_cache[i] = Tensor::create({_meta.maxseq, _meta.nkvh, _meta.dh}, _meta.dtype, _device_type, _device_id);
        _kv_cache.v_cache[i] = Tensor::create({_meta.maxseq, _meta.nkvh, _meta.dh}, _meta.dtype, _device_type, _device_id);
    }
}

void Qwen2Model::updateKVCache(size_t layer, tensor_t k, tensor_t v) {
    size_t new_len = k->shape()[0];
    for (size_t i = 0; i < new_len; ++i) {
        auto k_slice = k->slice(0, i, i + 1);
        auto v_slice = v->slice(0, i, i + 1);
        auto k_cache_slice = _kv_cache.k_cache[layer]->slice(0, _kv_cache.seq_len + i, _kv_cache.seq_len + i + 1);
        auto v_cache_slice = _kv_cache.v_cache[layer]->slice(0, _kv_cache.seq_len + i, _kv_cache.seq_len + i + 1);
        
        copy_data(k_cache_slice->data(), k_slice->data(), k_slice->numel() * k_slice->elementSize(), _device_type, LLAISYS_MEMCPY_D2D);
        copy_data(v_cache_slice->data(), v_slice->data(), v_slice->numel() * v_slice->elementSize(), _device_type, LLAISYS_MEMCPY_D2D);
    }
}

std::pair<tensor_t, tensor_t> Qwen2Model::getKVCache(size_t layer) {
    size_t total_len = _kv_cache.seq_len;
    if (total_len == 0) return {nullptr, nullptr};
    
    auto k = _kv_cache.k_cache[layer]->slice(0, 0, total_len);
    auto v = _kv_cache.v_cache[layer]->slice(0, 0, total_len);
    return {k, v};
}

tensor_t Qwen2Model::forward(tensor_t input_ids) {
    size_t seq_len = input_ids->shape()[0];
    
    // Embedding
    auto hidden = Tensor::create({seq_len, _meta.hs}, _meta.dtype, _device_type, _device_id);
    ops::embedding(hidden, input_ids, _weights.in_embed);
    
    // Create position ids
    auto pos_ids = Tensor::create({seq_len}, LLAISYS_DTYPE_I64, _device_type, _device_id);
    std::vector<int64_t> pos_data(seq_len);
    for (size_t i = 0; i < seq_len; ++i) {
        pos_data[i] = _kv_cache.seq_len + i;
    }
    pos_ids->load(pos_data.data());
    
    // Decoder layers
    for (size_t layer = 0; layer < _meta.nlayer; ++layer) {
        // Attention norm
        auto attn_norm_out = Tensor::create({seq_len, _meta.hs}, _meta.dtype, _device_type, _device_id);
        ops::rms_norm(attn_norm_out, hidden, _weights.attn_norm_w[layer], _meta.epsilon);
        
        // Q, K, V projections
        auto q = Tensor::create({seq_len, _meta.hs}, _meta.dtype, _device_type, _device_id);
        auto k = Tensor::create({seq_len, _meta.nkvh * _meta.dh}, _meta.dtype, _device_type, _device_id);
        auto v = Tensor::create({seq_len, _meta.nkvh * _meta.dh}, _meta.dtype, _device_type, _device_id);
        
        ops::linear(q, attn_norm_out, _weights.attn_q_w[layer], _weights.attn_q_b[layer]);
        ops::linear(k, attn_norm_out, _weights.attn_k_w[layer], _weights.attn_k_b[layer]);
        ops::linear(v, attn_norm_out, _weights.attn_v_w[layer], _weights.attn_v_b[layer]);
        
        // Reshape for multi-head attention
        q = q->view({seq_len, _meta.nh, _meta.dh});
        k = k->view({seq_len, _meta.nkvh, _meta.dh});
        v = v->view({seq_len, _meta.nkvh, _meta.dh});
        
        // Apply RoPE
        auto q_rope = Tensor::create({seq_len, _meta.nh, _meta.dh}, _meta.dtype, _device_type, _device_id);
        auto k_rope = Tensor::create({seq_len, _meta.nkvh, _meta.dh}, _meta.dtype, _device_type, _device_id);
        ops::rope(q_rope, q, pos_ids, _meta.theta);
        ops::rope(k_rope, k, pos_ids, _meta.theta);
        
        // Update KV cache
        updateKVCache(layer, k_rope, v);
        
        // Get full K, V from cache
        auto [k_cache, v_cache] = getKVCache(layer);
        
        // Concatenate with current K, V
        size_t total_seq_len = _kv_cache.seq_len + seq_len;
        auto k_full = Tensor::create({total_seq_len, _meta.nkvh, _meta.dh}, _meta.dtype, _device_type, _device_id);
        auto v_full = Tensor::create({total_seq_len, _meta.nkvh, _meta.dh}, _meta.dtype, _device_type, _device_id);
        
        if (k_cache) {
            copy_data(k_full->data(), k_cache->data(), k_cache->numel() * k_cache->elementSize(), _device_type, LLAISYS_MEMCPY_D2D);
            copy_data(v_full->data(), v_cache->data(), v_cache->numel() * v_cache->elementSize(), _device_type, LLAISYS_MEMCPY_D2D);
            for (size_t i = 0; i < seq_len; ++i) {
                auto k_rope_slice = k_rope->slice(0, i, i + 1);
                auto v_slice = v->slice(0, i, i + 1);
                auto k_full_slice = k_full->slice(0, _kv_cache.seq_len + i, _kv_cache.seq_len + i + 1);
                auto v_full_slice = v_full->slice(0, _kv_cache.seq_len + i, _kv_cache.seq_len + i + 1);
                copy_data(k_full_slice->data(), k_rope_slice->data(), k_rope_slice->numel() * k_rope_slice->elementSize(), _device_type, LLAISYS_MEMCPY_D2D);
                copy_data(v_full_slice->data(), v_slice->data(), v_slice->numel() * v_slice->elementSize(), _device_type, LLAISYS_MEMCPY_D2D);
            }
        } else {
            copy_data(k_full->data(), k_rope->data(), k_rope->numel() * k_rope->elementSize(), _device_type, LLAISYS_MEMCPY_D2D);
            copy_data(v_full->data(), v->data(), v->numel() * v->elementSize(), _device_type, LLAISYS_MEMCPY_D2D);
        }
        
        // Self attention
        float scale = 1.0f / std::sqrt(static_cast<float>(_meta.dh));
        auto attn_out = Tensor::create({seq_len, _meta.nh, _meta.dh}, _meta.dtype, _device_type, _device_id);
        ops::self_attention(attn_out, q_rope, k_full, v_full, scale);
        
        // Reshape and project
        attn_out = attn_out->view({seq_len, _meta.hs});
        auto attn_proj = Tensor::create({seq_len, _meta.hs}, _meta.dtype, _device_type, _device_id);
        ops::linear(attn_proj, attn_out, _weights.attn_o_w[layer], nullptr);
        
        // Residual
        auto hidden_after_attn = Tensor::create({seq_len, _meta.hs}, _meta.dtype, _device_type, _device_id);
        ops::add(hidden_after_attn, hidden, attn_proj);
        
        // MLP norm
        auto mlp_norm_out = Tensor::create({seq_len, _meta.hs}, _meta.dtype, _device_type, _device_id);
        ops::rms_norm(mlp_norm_out, hidden_after_attn, _weights.mlp_norm_w[layer], _meta.epsilon);
        
        // MLP
        auto gate = Tensor::create({seq_len, _meta.di}, _meta.dtype, _device_type, _device_id);
        auto up = Tensor::create({seq_len, _meta.di}, _meta.dtype, _device_type, _device_id);
        ops::linear(gate, mlp_norm_out, _weights.mlp_gate_w[layer], nullptr);
        ops::linear(up, mlp_norm_out, _weights.mlp_up_w[layer], nullptr);
        
        auto swiglu_out = Tensor::create({seq_len, _meta.di}, _meta.dtype, _device_type, _device_id);
        ops::swiglu(swiglu_out, gate, up);
        
        auto mlp_out = Tensor::create({seq_len, _meta.hs}, _meta.dtype, _device_type, _device_id);
        ops::linear(mlp_out, swiglu_out, _weights.mlp_down_w[layer], nullptr);
        
        // Residual
        hidden = Tensor::create({seq_len, _meta.hs}, _meta.dtype, _device_type, _device_id);
        ops::add(hidden, hidden_after_attn, mlp_out);
    }
    
    // Update cache length
    _kv_cache.seq_len += seq_len;
    
    // Final norm
    auto final_norm = Tensor::create({seq_len, _meta.hs}, _meta.dtype, _device_type, _device_id);
    ops::rms_norm(final_norm, hidden, _weights.out_norm_w, _meta.epsilon);
    
    // Output projection
    auto logits = Tensor::create({seq_len, _meta.voc}, _meta.dtype, _device_type, _device_id);
    ops::linear(logits, final_norm, _weights.out_embed, nullptr);
    
    return logits;
}

int64_t Qwen2Model::infer(const std::vector<int64_t> &token_ids, float temperature, float top_p, int top_k) {
    auto input_ids = Tensor::create({token_ids.size()}, LLAISYS_DTYPE_I64, _device_type, _device_id);
    input_ids->load(token_ids.data());
    
    auto logits = forward(input_ids);
    
    // Get last token logits
    auto last_logits = logits->slice(0, logits->shape()[0] - 1, logits->shape()[0]);
    last_logits = last_logits->view({_meta.voc});
    
    std::vector<float> logits_f32(_meta.voc);
    if (_meta.dtype == LLAISYS_DTYPE_F32) {
        copy_data(logits_f32.data(), last_logits->data(), _meta.voc * sizeof(float), _device_type, LLAISYS_MEMCPY_D2H);
    } else {
        // Implement dtype conversion logic here if needed
        return -1;
    }

    if (temperature <= 0.0f) {
        // Greedy search
        auto max_it = std::max_element(logits_f32.begin(), logits_f32.end());
        return std::distance(logits_f32.begin(), max_it);
    }

    // Temperature scaling
    for (float& val : logits_f32) {
        val /= temperature;
    }

    // Softmax
    float max_logit = *std::max_element(logits_f32.begin(), logits_f32.end());
    float sum_exp = 0.0f;
    for (float& val : logits_f32) {
        val = std::exp(val - max_logit);
        sum_exp += val;
    }
    for (float& val : logits_f32) {
        val /= sum_exp;
    }

    // Top-k filtering
    std::vector<std::pair<float, int>> probs;
    probs.reserve(_meta.voc);
    for (size_t i = 0; i < _meta.voc; ++i) {
        probs.emplace_back(logits_f32[i], i);
    }

    if (top_k > 0 && top_k < (int)_meta.voc) {
        std::partial_sort(probs.begin(), probs.begin() + top_k, probs.end(),
                          [](const auto& a, const auto& b) { return a.first > b.first; });
        probs.resize(top_k);
    } else {
        std::sort(probs.begin(), probs.end(),
                  [](const auto& a, const auto& b) { return a.first > b.first; });
    }

    // Top-p filtering
    float cumulative_prob = 0.0f;
    size_t top_p_idx = probs.size();
    for (size_t i = 0; i < probs.size(); ++i) {
        cumulative_prob += probs[i].first;
        if (cumulative_prob >= top_p) {
            top_p_idx = i + 1;
            break;
        }
    }
    probs.resize(top_p_idx);

    // Re-normalize after top-p
    sum_exp = 0.0f;
    for (const auto& p : probs) {
        sum_exp += p.first;
    }
    
    std::vector<float> final_probs;
    final_probs.reserve(probs.size());
    for (const auto& p : probs) {
        final_probs.push_back(p.first / sum_exp);
    }

    // Sample
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::discrete_distribution<int> dist(final_probs.begin(), final_probs.end());
    
    return probs[dist(gen)].second;
}

} // namespace llaisys::models
