#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/argmax_cpu.hpp"
#include "nvidia/argmax_nvidia.cuh"
#include "musa/argmax_musa.muh"



namespace llaisys::ops {
void argmax(tensor_t max_idx, tensor_t max_val, tensor_t vals) {
    // 基本合法性检查：设备、dtype、contiguous 等
    CHECK_SAME_DEVICE(max_idx, max_val, vals);
    CHECK_SAME_DTYPE(max_val->dtype(), vals->dtype());
    ASSERT(max_idx->isContiguous() && max_val->isContiguous() && vals->isContiguous(),
           "Argmax: all tensors must be contiguous.");

    // TODO: 当前假设对 vals 的所有元素做一维全局 argmax，输出标量 max_val / max_idx。
    // 如果以后需要支持按某个维度 argmax 或保持形状，可以在这里扩展逻辑。


    if (vals->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::argmax(max_idx->data(),
                           max_val->data(),
                           vals->data(),
                           vals->dtype(),
                           max_idx->dtype(),
                           vals->numel());
    }

    llaisys::core::context().setDevice(vals->deviceType(), vals->deviceId());

    switch (vals->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::argmax(max_idx->data(),
                           max_val->data(),
                           vals->data(),
                           vals->dtype(),
                           max_idx->dtype(),
                           vals->numel());
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        nvidia::argmax(max_idx->data(),
                           max_val->data(),
                           vals->data(),
                           vals->dtype(),
                           max_idx->dtype(),
                           vals->numel()
        );
        return;
#endif
#ifdef ENABLE_MUSA_API
    case LLAISYS_DEVICE_MUSA:
        musa::argmax(max_idx->data(),
                           max_val->data(),
                           vals->data(),
                           vals->dtype(),
                           max_idx->dtype(),
                           vals->numel()
        );
        return;
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
