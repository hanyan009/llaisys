#include "llaisys/models/qwen2.h"
#include "../models/qwen2/qwen2_model.hpp"
#include "llaisys_tensor.hpp"
#include <cstring>

using namespace llaisys::models;

struct LlaisysQwen2Model {
    Qwen2Model *model;
};

struct LlaisysQwen2Model *llaisysQwen2ModelCreate(const LlaisysQwen2Meta *meta, llaisysDeviceType_t device, int *device_ids, int ndevice) {
    Qwen2Meta cpp_meta;
    cpp_meta.dtype = meta->dtype;
    cpp_meta.nlayer = meta->nlayer;
    cpp_meta.hs = meta->hs;
    cpp_meta.nh = meta->nh;
    cpp_meta.nkvh = meta->nkvh;
    cpp_meta.dh = meta->dh;
    cpp_meta.di = meta->di;
    cpp_meta.maxseq = meta->maxseq;
    cpp_meta.voc = meta->voc;
    cpp_meta.epsilon = meta->epsilon;
    cpp_meta.theta = meta->theta;
    cpp_meta.end_token = meta->end_token;

    int device_id = (ndevice > 0 && device_ids) ? device_ids[0] : 0;
    
    auto *wrapper = new LlaisysQwen2Model();
    wrapper->model = new Qwen2Model(cpp_meta, device, device_id);
    return wrapper;
}

void llaisysQwen2ModelDestroy(struct LlaisysQwen2Model *model) {
    if (model) {
        delete model->model;
        delete model;
    }
}

struct LlaisysQwen2Weights *llaisysQwen2ModelWeights(struct LlaisysQwen2Model *model) {
    if (!model || !model->model) return nullptr;
    
    auto &weights = model->model->weights();
    auto *c_weights = new LlaisysQwen2Weights();
    
    printf("[DEBUG] in_embed tensor pointer: %p\n", weights.in_embed.get());
    printf("[DEBUG] in_embed shared_ptr use_count: %ld\n", weights.in_embed.use_count());
    if (weights.in_embed) {
        printf("[DEBUG] in_embed ndim: %zu\n", weights.in_embed->ndim());
        auto shape = weights.in_embed->shape();
        printf("[DEBUG] in_embed shape: [");
        for (size_t i = 0; i < shape.size(); ++i) {
            printf("%zu", shape[i]);
            if (i < shape.size() - 1) printf(", ");
        }
        printf("]\n");
    }
    
    // Create LlaisysTensor wrappers (these need to persist!)
    auto *in_embed_wrapper = new LlaisysTensor{weights.in_embed};
    auto *out_embed_wrapper = new LlaisysTensor{weights.out_embed};
    auto *out_norm_w_wrapper = new LlaisysTensor{weights.out_norm_w};
    
    c_weights->in_embed = in_embed_wrapper;
    c_weights->out_embed = out_embed_wrapper;
    c_weights->out_norm_w = out_norm_w_wrapper;
    
    size_t nlayer = weights.attn_norm_w.size();
    c_weights->attn_norm_w = new llaisysTensor_t[nlayer];
    c_weights->attn_q_w = new llaisysTensor_t[nlayer];
    c_weights->attn_q_b = new llaisysTensor_t[nlayer];
    c_weights->attn_k_w = new llaisysTensor_t[nlayer];
    c_weights->attn_k_b = new llaisysTensor_t[nlayer];
    c_weights->attn_v_w = new llaisysTensor_t[nlayer];
    c_weights->attn_v_b = new llaisysTensor_t[nlayer];
    c_weights->attn_o_w = new llaisysTensor_t[nlayer];
    c_weights->mlp_norm_w = new llaisysTensor_t[nlayer];
    c_weights->mlp_gate_w = new llaisysTensor_t[nlayer];
    c_weights->mlp_up_w = new llaisysTensor_t[nlayer];
    c_weights->mlp_down_w = new llaisysTensor_t[nlayer];
    
    for (size_t i = 0; i < nlayer; ++i) {
        c_weights->attn_norm_w[i] = new LlaisysTensor{weights.attn_norm_w[i]};
        c_weights->attn_q_w[i] = new LlaisysTensor{weights.attn_q_w[i]};
        c_weights->attn_q_b[i] = new LlaisysTensor{weights.attn_q_b[i]};
        c_weights->attn_k_w[i] = new LlaisysTensor{weights.attn_k_w[i]};
        c_weights->attn_k_b[i] = new LlaisysTensor{weights.attn_k_b[i]};
        c_weights->attn_v_w[i] = new LlaisysTensor{weights.attn_v_w[i]};
        c_weights->attn_v_b[i] = new LlaisysTensor{weights.attn_v_b[i]};
        c_weights->attn_o_w[i] = new LlaisysTensor{weights.attn_o_w[i]};
        c_weights->mlp_norm_w[i] = new LlaisysTensor{weights.mlp_norm_w[i]};
        c_weights->mlp_gate_w[i] = new LlaisysTensor{weights.mlp_gate_w[i]};
        c_weights->mlp_up_w[i] = new LlaisysTensor{weights.mlp_up_w[i]};
        c_weights->mlp_down_w[i] = new LlaisysTensor{weights.mlp_down_w[i]};
    }
    
    return c_weights;
}

int64_t llaisysQwen2ModelInfer(struct LlaisysQwen2Model *model, int64_t *token_ids, size_t ntoken, float temperature, float top_p, int top_k) {
    if (!model || !model->model) return -1;
    
    std::vector<int64_t> tokens(token_ids, token_ids + ntoken);
    return model->model->infer(tokens, temperature, top_p, top_k);
}
