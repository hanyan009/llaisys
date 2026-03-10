#include "nvidia_resource.cuh"
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <unordered_map>
#include <memory>

#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

#define CHECK_CUBLAS(call) \
    do { \
        cublasStatus_t status = call; \
        if (status != CUBLAS_STATUS_SUCCESS) { \
            fprintf(stderr, "CUBLAS error at %s:%d: %d\n", __FILE__, __LINE__, \
                    status); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

#define CHECK_CUDNN(call) \
    do { \
        cudnnStatus_t status = call; \
        if (status != CUDNN_STATUS_SUCCESS) { \
            fprintf(stderr, "CUDNN error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudnnGetErrorString(status)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

namespace llaisys::device::nvidia {

Resource::Resource(int device_id) : llaisys::device::DeviceResource(LLAISYS_DEVICE_NVIDIA, device_id) {
    CHECK_CUDA(cudaSetDevice(device_id));
    CHECK_CUBLAS(cublasCreate(&cublas_handle_));
    CHECK_CUDNN(cudnnCreate(&cudnn_handle_));
}

Resource::~Resource() {
    // destructors should not throw exceptions or exit, so we just log errors
    cublasStatus_t cublas_err = cublasDestroy(cublas_handle_);
    if (cublas_err != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "CUBLAS destroy error: %d\n", cublas_err);
    }

    cudnnStatus_t cudnn_err = cudnnDestroy(cudnn_handle_);
    if (cudnn_err != CUDNN_STATUS_SUCCESS) {
        fprintf(stderr, "CUDNN destroy error: %s\n", cudnnGetErrorString(cudnn_err));
    }
}

static std::mutex g_resource_mutex;
static std::unordered_map<int, std::unique_ptr<Resource>> g_resources;

Resource* get_resource(int device_id) {
    std::lock_guard<std::mutex> lock(g_resource_mutex);
    auto it = g_resources.find(device_id);
    if (it == g_resources.end()) {
        g_resources[device_id] = std::make_unique<Resource>(device_id);
        return g_resources[device_id].get();
    }
    return it->second.get();
}

} // namespace llaisys::device::nvidia
