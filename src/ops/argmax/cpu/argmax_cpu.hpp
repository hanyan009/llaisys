#pragma once
#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::cpu {

// CPU 版本的 argmax 内核函数，仅负责对原始内存做计算，不关心 tensor 封装。
// 参数说明：
// - max_idx: 输出最大值位置的缓冲区指针（例如 i64），与 max_val 在同一设备上；
// - max_val: 输出最大值的缓冲区指针，数据类型与 vals 相同；
// - vals:    输入数据缓冲区指针，按一维展平；
// - val_type: 输入/输出值的元素数据类型（例如 f32/f16/bf16）；
// - idx_type: 索引的元素数据类型（当前测试使用 i64）；
// - numel:   输入 vals 中的元素个数（当前测试用例是一维向量的长度）。
//
// TODO: 在 cpp 中根据 val_type / idx_type 做类型分发，并实现一维全局 argmax。
void argmax(std::byte *max_idx,
            std::byte *max_val,
            const std::byte *vals,
            llaisysDataType_t val_type,
            llaisysDataType_t idx_type,
            size_t numel);

} // namespace llaisys::ops::cpu
