#include "embedding_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>

template <typename T, typename IndexT>
void embedding_(T *out, const IndexT *index, const T *weight, size_t batch_size, size_t hidden_size) {
    for (size_t i = 0; i < batch_size; i++) {
        IndexT idx = index[i];
        const T *weight_row = weight + idx * hidden_size;
        T *out_row = out + i * hidden_size;
        for (size_t j = 0; j < hidden_size; j++) {
            out_row[j] = weight_row[j];
        }
    }
}

template <typename T>
void embedding_dispatch_index(T *out, const std::byte *index, const T *weight, llaisysDataType_t index_type, size_t batch_size,
                               size_t hidden_size) {
    switch (index_type) {
    case LLAISYS_DTYPE_I32:
        return embedding_(out, reinterpret_cast<const int32_t *>(index), weight, batch_size, hidden_size);
    case LLAISYS_DTYPE_I64:
        return embedding_(out, reinterpret_cast<const int64_t *>(index), weight, batch_size, hidden_size);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(index_type);
    }
}

namespace llaisys::ops::cpu {
void embedding(std::byte *out, const std::byte *index, const std::byte *weight, llaisysDataType_t out_type,
               llaisysDataType_t index_type, size_t batch_size, size_t hidden_size) {
    switch (out_type) {
    case LLAISYS_DTYPE_F32:
        return embedding_dispatch_index(reinterpret_cast<float *>(out), index, reinterpret_cast<const float *>(weight), index_type,
                                        batch_size, hidden_size);
    case LLAISYS_DTYPE_BF16:
        return embedding_dispatch_index(reinterpret_cast<llaisys::bf16_t *>(out), index,
                                        reinterpret_cast<const llaisys::bf16_t *>(weight), index_type, batch_size, hidden_size);
    case LLAISYS_DTYPE_F16:
        return embedding_dispatch_index(reinterpret_cast<llaisys::fp16_t *>(out), index,
                                        reinterpret_cast<const llaisys::fp16_t *>(weight), index_type, batch_size, hidden_size);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(out_type);
    }
}
} // namespace llaisys::ops::cpu
