#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"
#include "nvidia/rearrange_nvidia.cuh"
#include "musa/rearrange_musa.muh"

namespace llaisys::ops {
void rearrange(tensor_t out, tensor_t in) {
    CHECK_SAME_DEVICE(out, in);
    CHECK_SAME_SHAPE(out->shape(), in->shape());
    CHECK_SAME_DTYPE(out->dtype(), in->dtype());
    
    // out must be contiguous
    ASSERT(out->isContiguous(), "Rearrange: output must be contiguous.");

    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        TO_BE_IMPLEMENTED(); 
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        TO_BE_IMPLEMENTED();
        return;
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return llaisys::ops::nvidia::rearrange(out, in);
#endif
#ifdef ENABLE_MUSA_API
    case LLAISYS_DEVICE_MUSA:
        return llaisys::ops::musa::rearrange(out, in);
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
