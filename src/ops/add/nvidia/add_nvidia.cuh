#pragma once

#ifdef ENABLE_NVIDIA_API
namespace llaisys::ops::nvidia {
void add(llaisys::tensor_t c, llaisys::tensor_t a, llaisys::tensor_t b);
}
#endif