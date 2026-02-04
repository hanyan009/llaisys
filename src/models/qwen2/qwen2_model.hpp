#pragma once
#include "../../tensor/tensor.hpp"
#include "../../ops/add/op.hpp"
#include "../../ops/embedding/op.hpp"
#include "../../ops/linear/op.hpp"
#include "../../ops/rms_norm/op.hpp"
#include "../../ops/rope/op.hpp"
#include "../../ops/self_attention/op.hpp"
#include "../../ops/swiglu/op.hpp"
#include "../../ops/argmax/op.hpp"
#include "../../ops/rearrange/op.hpp"
#include <vector>

namespace llaisys::models {

struct Qwen2Meta {
    llaisysDataType_t dtype;
    size_t nlayer, hs, nh, nkvh, dh, di, maxseq, voc;
    float epsilon, theta;
    int64_t end_token;
};

struct Qwen2Weights {
    tensor_t in_embed;
    tensor_t out_embed;
    tensor_t out_norm_w;
    std::vector<tensor_t> attn_norm_w;
    std::vector<tensor_t> attn_q_w;
    std::vector<tensor_t> attn_q_b;
    std::vector<tensor_t> attn_k_w;
    std::vector<tensor_t> attn_k_b;
    std::vector<tensor_t> attn_v_w;
    std::vector<tensor_t> attn_v_b;
    std::vector<tensor_t> attn_o_w;
    std::vector<tensor_t> mlp_norm_w;
    std::vector<tensor_t> mlp_gate_w;
    std::vector<tensor_t> mlp_up_w;
    std::vector<tensor_t> mlp_down_w;
};

struct KVCache {
    std::vector<tensor_t> k_cache;
    std::vector<tensor_t> v_cache;
    size_t seq_len;
};

class Qwen2Model {
private:
    Qwen2Meta _meta;
    Qwen2Weights _weights;
    KVCache _kv_cache;
    llaisysDeviceType_t _device_type;
    int _device_id;

public:
    Qwen2Model(const Qwen2Meta &meta, llaisysDeviceType_t device_type, int device_id);
    ~Qwen2Model() = default;

    Qwen2Weights &weights() { return _weights; }
    int64_t infer(const std::vector<int64_t> &token_ids);

private:
    tensor_t forward(tensor_t input_ids);
    void initKVCache();
    void updateKVCache(size_t layer, tensor_t k, tensor_t v);
    std::pair<tensor_t, tensor_t> getKVCache(size_t layer);
};

} // namespace llaisys::models
