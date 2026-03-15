#include "musa_resource.muh"
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <unordered_map>
#include <memory>
#include <musa_runtime.h>

#define CHECK_MUSA(call) \
    do { \
        musaError_t err = call; \
        if (err != musaSuccess) { \
            fprintf(stderr, "MUSA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    musaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

#define CHECK_MUBLAS(call) \
    do { \
        mublasStatus_t status = call; \
        if (status != MUBLAS_STATUS_SUCCESS) { \
            fprintf(stderr, "MUBLAS error at %s:%d: %d\n", __FILE__, __LINE__, \
                    status); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

namespace llaisys::device::musa {

Resource::Resource(int device_id) : llaisys::device::DeviceResource(LLAISYS_DEVICE_MUSA, device_id) {
    CHECK_MUSA(musaSetDevice(device_id));
    CHECK_MUBLAS(mublasCreate(&mublas_handle_));
}

Resource::~Resource() {
    mublasStatus_t mublas_err = mublasDestroy(mublas_handle_);
    if (mublas_err != MUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "MUBLAS destroy error: %d\n", mublas_err);
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

} // namespace llaisys::device::musa
