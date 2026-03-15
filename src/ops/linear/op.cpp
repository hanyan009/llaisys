#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/linear_cpu.hpp"
#include "nvidia/linear_nvidia.cuh"
#include "musa/linear_musa.muh"

namespace llaisys::ops {
void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias) {
    CHECK_SAME_DEVICE(out, in, weight);
    // in: [batch_size, in_features]
    // weight: [out_features, in_features]
    // bias: [out_features] or nullptr
    // out: [batch_size, out_features]
    ASSERT(out->isContiguous() && in->isContiguous() && weight->isContiguous(), "Linear: all tensors must be contiguous.");
    ASSERT(in->ndim() == 2, "Linear: input must be 2D tensor.");
    ASSERT(weight->ndim() == 2, "Linear: weight must be 2D tensor.");
    ASSERT(out->dtype() == in->dtype() && in->dtype() == weight->dtype(), "Linear: all tensors must have same dtype.");

    bool has_bias = (bias != nullptr);
    if (has_bias) {
        CHECK_SAME_DEVICE(out, bias);
        ASSERT(bias->isContiguous(), "Linear: bias must be contiguous.");
        ASSERT(bias->ndim() == 1, "Linear: bias must be 1D tensor.");
        ASSERT(bias->dtype() == out->dtype(), "Linear: bias must have same dtype as output.");
    }

    size_t batch_size = in->shape()[0];
    size_t in_features = in->shape()[1];
    size_t out_features = weight->shape()[0];

    ASSERT(weight->shape()[1] == in_features, "Linear: weight shape mismatch.");
    if (has_bias) {
        ASSERT(bias->shape()[0] == out_features, "Linear: bias shape mismatch.");
    }

    // always support cpu calculation
    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::linear(out->data(), in->data(), weight->data(), has_bias ? bias->data() : nullptr, out->dtype(), batch_size,
                           in_features, out_features, has_bias);
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::linear(out->data(), in->data(), weight->data(), has_bias ? bias->data() : nullptr, out->dtype(), batch_size,
                           in_features, out_features, has_bias);
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return llaisys::ops::nvidia::linear(out, in, weight, bias);
#endif
#ifdef ENABLE_MUSA_API
    case LLAISYS_DEVICE_MUSA:
        return llaisys::ops::musa::linear(out, in, weight, bias);
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
