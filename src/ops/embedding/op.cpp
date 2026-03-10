#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/embedding_cpu.hpp"
#include "nvidia/embedding_nvidia.cuh"

namespace llaisys::ops {
void embedding(tensor_t out, tensor_t index, tensor_t weight) {
    CHECK_SAME_DEVICE(out, index, weight);
    // index: [batch_size] or [batch_size, seq_len]
    // weight: [vocab_size, hidden_size]
    // out: [batch_size, hidden_size] or [batch_size, seq_len, hidden_size]
    ASSERT(out->isContiguous() && index->isContiguous() && weight->isContiguous(),
           "Embedding: all tensors must be contiguous.");
    ASSERT(weight->ndim() == 2, "Embedding: weight must be 2D tensor.");
    ASSERT(out->dtype() == weight->dtype(), "Embedding: out and weight must have same dtype.");

    size_t batch_size = index->numel();
    size_t hidden_size = weight->shape()[1];

    // always support cpu calculation
    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::embedding(out->data(), index->data(), weight->data(), out->dtype(), index->dtype(), batch_size, hidden_size);
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::embedding(out->data(), index->data(), weight->data(), out->dtype(), index->dtype(), batch_size, hidden_size);
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return llaisys::ops::nvidia::embedding(out, index, weight);
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
