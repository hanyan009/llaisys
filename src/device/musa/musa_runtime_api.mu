#include "../runtime_api.hpp"

#include <cstdlib>
#include <cstring>
#include <musa_runtime.h>
#include <cstdio>

#define MUSA_CHECK(condition) \
  do { \
    musaError_t error = condition; \
    if (error != musaSuccess) { \
      printf("MUSA Error: %s at %s:%d\n", musaGetErrorString(error), __FILE__, __LINE__); \
      exit(EXIT_FAILURE); \
    } \
  } while (0)

namespace llaisys::device::musa {

namespace runtime_api {
int getDeviceCount() {
    int count = 0;
    musaError_t error = musaGetDeviceCount(&count);
    if (error != musaSuccess) {
        printf("musaGetDeviceCount failed: %s\n", musaGetErrorString(error));
        return 0;
    }
    return count;
}

void setDevice(int device) {
    MUSA_CHECK(musaSetDevice(device));
}

void deviceSynchronize() {
    MUSA_CHECK(musaDeviceSynchronize());
}

llaisysStream_t createStream() {
    musaStream_t stream;
    MUSA_CHECK(musaStreamCreate(&stream));
    return (llaisysStream_t)stream;
}

void destroyStream(llaisysStream_t stream) {
    MUSA_CHECK(musaStreamDestroy((musaStream_t)stream));
}
void streamSynchronize(llaisysStream_t stream) {
    MUSA_CHECK(musaStreamSynchronize((musaStream_t)stream));
}

void *mallocDevice(size_t size) {
    void *ptr;
    MUSA_CHECK(musaMalloc(&ptr, size));
    return ptr;
}

void freeDevice(void *ptr) {
    MUSA_CHECK(musaFree(ptr));
}

void *mallocHost(size_t size) {
    void *ptr;
    MUSA_CHECK(musaMallocHost(&ptr, size));
    return ptr;
}

void freeHost(void *ptr) {
    MUSA_CHECK(musaFreeHost(ptr));
}

musaMemcpyKind getMusaMemcpyKind(llaisysMemcpyKind_t kind) {
    switch (kind) {
    case LLAISYS_MEMCPY_H2H:
        return musaMemcpyHostToHost;
    case LLAISYS_MEMCPY_H2D:
        return musaMemcpyHostToDevice;
    case LLAISYS_MEMCPY_D2H:
        return musaMemcpyDeviceToHost;
    case LLAISYS_MEMCPY_D2D:
        return musaMemcpyDeviceToDevice;
    default:
        return musaMemcpyDefault;
    }
}

void memcpySync(void *dst, const void *src, size_t size, llaisysMemcpyKind_t kind) {
    MUSA_CHECK(musaMemcpy(dst, src, size, getMusaMemcpyKind(kind)));
}

void memcpyAsync(void *dst, const void *src, size_t size, llaisysMemcpyKind_t kind, llaisysStream_t stream) {
    MUSA_CHECK(musaMemcpyAsync(dst, src, size, getMusaMemcpyKind(kind), (musaStream_t)stream));
}

static const LlaisysRuntimeAPI RUNTIME_API = {
    &getDeviceCount,
    &setDevice,
    &deviceSynchronize,
    &createStream,
    &destroyStream,
    &streamSynchronize,
    &mallocDevice,
    &freeDevice,
    &mallocHost,
    &freeHost,
    &memcpySync,
    &memcpyAsync};

} // namespace runtime_api

const LlaisysRuntimeAPI *getRuntimeAPI() {
    return &runtime_api::RUNTIME_API;
}
} // namespace llaisys::device::musa
