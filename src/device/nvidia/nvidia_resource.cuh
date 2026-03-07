#pragma once

#include "../device_resource.hpp"

#include <cublas_v2.h>
#include <cudnn.h>

namespace llaisys::device::nvidia {
class Resource : public llaisys::device::DeviceResource {
public:
    Resource(int device_id);
    ~Resource();

    cublasHandle_t cublas_handle() { return cublas_handle_; }
    cudnnHandle_t cudnn_handle() { return cudnn_handle_; }

private:
    cublasHandle_t cublas_handle_;
    cudnnHandle_t cudnn_handle_;
};
} // namespace llaisys::device::nvidia
