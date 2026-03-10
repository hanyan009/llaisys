#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"
#include "nvidia/rearrange_nvidia.cuh"

namespace llaisys::ops {
void rearrange(tensor_t out, tensor_t in) {
    CHECK_SAME_DEVICE(out, in);
    CHECK_SAME_SHAPE(out->shape(), in->shape());
    CHECK_SAME_DTYPE(out->dtype(), in->dtype());
    
    // out must be contiguous
    ASSERT(out->isContiguous(), "Rearrange: output must be contiguous.");

    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        // CPU implementation: simple copy with stride handling
        // Since we don't have a dedicated CPU kernel file yet, and this is "bonus",
        // we can implement a simple loop here or use a helper.
        // For now, let's assume the user only cares about CUDA as per instruction, 
        // but for CPU fallback we might need something.
        // The README says "Task-2.9 (Optional) rearrange ... used to copy data ... implementation the cpu version".
        // I will assume CPU is not my priority right now unless I need it.
        // But wait, the existing code was empty TO_BE_IMPLEMENTED.
        // I will implement a basic CPU version to avoid crashes if called on CPU.
        
        // Actually, let's stick to the requested CUDA implementation first.
        // But the function signature in existing op.cpp needs to be replaced.
        
        // Let's implement a generic CPU walker or just leave it if I'm only doing CUDA.
        // The prompt asked for "Implement CUDA operators".
        // I will just add the dispatch.
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
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
