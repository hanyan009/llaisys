#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/rope_cpu.hpp"
#include "nvidia/rope_nvidia.cuh"

namespace llaisys::ops {
void rope(tensor_t out, tensor_t in, tensor_t pos_ids, float theta) {
    CHECK_SAME_DEVICE(out, in, pos_ids);
    CHECK_SAME_DTYPE(out->dtype(), in->dtype());
    ASSERT(out->isContiguous() && in->isContiguous() && pos_ids->isContiguous(),
           "Rope: all tensors must be contiguous.");
    
    // out and in should have shape [seq_len, n_heads, head_dim]
    ASSERT(out->ndim() == 3 && in->ndim() == 3, "Rope: out and in must be 3D tensors.");
    CHECK_SAME_SHAPE(out->shape(), in->shape());
    
    size_t seq_len = out->shape()[0];
    size_t n_heads = out->shape()[1];
    size_t head_dim = out->shape()[2];
    
    // pos_ids should be 1D with length seq_len
    ASSERT(pos_ids->ndim() == 1, "Rope: pos_ids must be 1D tensor.");
    ASSERT(pos_ids->shape()[0] == seq_len, "Rope: pos_ids length must match seq_len.");
    ASSERT(pos_ids->dtype() == LLAISYS_DTYPE_I64, "Rope: pos_ids must be int64.");
    ASSERT(head_dim % 2 == 0, "Rope: head_dim must be even.");

    // always support cpu calculation
    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::rope(out->data(), in->data(), pos_ids->data(), out->dtype(),
                         seq_len, n_heads, head_dim, theta);
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::rope(out->data(), in->data(), pos_ids->data(), out->dtype(),
                         seq_len, n_heads, head_dim, theta);
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return llaisys::ops::nvidia::rope(out, in, pos_ids, theta);
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
