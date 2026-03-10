#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/self_attention_cpu.hpp"
#include "nvidia/self_attention_nvidia.cuh"

namespace llaisys::ops {
void self_attention(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale) {
    CHECK_SAME_DEVICE(attn_val, q, k, v);
    CHECK_SAME_DTYPE(attn_val->dtype(), q->dtype(), k->dtype(), v->dtype());
    ASSERT(attn_val->isContiguous() && q->isContiguous() && k->isContiguous() && v->isContiguous(),
           "Self_attention: all tensors must be contiguous.");
    
    // q, k, v should be 3D tensors: [seq_len, num_heads, head_dim]
    ASSERT(q->ndim() == 3 && k->ndim() == 3 && v->ndim() == 3 && attn_val->ndim() == 3,
           "Self_attention: q, k, v, attn_val must be 3D tensors.");
    
    size_t qlen = q->shape()[0];
    size_t nh = q->shape()[1];      // number of query heads
    size_t hd = q->shape()[2];      // head dimension
    
    size_t kvlen = k->shape()[0];
    size_t nkvh = k->shape()[1];    // number of key/value heads
    
    // Validate shapes
    ASSERT(k->shape()[2] == hd && v->shape()[2] == hd,
           "Self_attention: head dimension must match.");
    CHECK_SAME_SHAPE(k->shape(), v->shape());
    ASSERT(attn_val->shape()[0] == qlen && attn_val->shape()[1] == nh && attn_val->shape()[2] == hd,
           "Self_attention: attn_val shape must match q shape.");
    ASSERT(nh % nkvh == 0, "Self_attention: number of query heads must be divisible by number of kv heads.");

    // always support cpu calculation
    if (attn_val->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::self_attention(attn_val->data(), q->data(), k->data(), v->data(),
                                   attn_val->dtype(), qlen, kvlen, nh, nkvh, hd, scale);
    }

    llaisys::core::context().setDevice(attn_val->deviceType(), attn_val->deviceId());

    switch (attn_val->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::self_attention(attn_val->data(), q->data(), k->data(), v->data(),
                                   attn_val->dtype(), qlen, kvlen, nh, nkvh, hd, scale);
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return llaisys::ops::nvidia::self_attention(attn_val, q, k, v, scale);
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
