#pragma once
#include "../core/llaisys_core.hpp"

#include <vector>
namespace llaisys {
class Tensor;
using tensor_t = std::shared_ptr<Tensor>;

struct TensorMeta {
    llaisysDataType_t dtype;
    std::vector<size_t> shape; // size_t 一般用来表示最大的容量，是无符号整型变量。（_t后缀表示类型，是一种约定）
    std::vector<ptrdiff_t> strides; // ptrdiff_t 一般用来计算两个指针之间的距离，是有符号整型变量。
};

class Tensor {
private:
    TensorMeta _meta; // 存储张量的数据类型、形状、步长参数
    core::storage_t _storage; // 存储张量的数据本身。【TODO】还需要深入去看一看其中的API
    size_t _offset; // _offset 表示张量在内存中的起始地址。
    // 【问】为什么 _offset 用 size_t 类型？
    // 【答】他是无符号整型。为什么 offset_ 用 size_t 类型? 因为他是64位无符号整型，符号上来说非负。容量上来说64位表达的寻址空间足够大，16EB（TB→PB→EB）。
    Tensor(TensorMeta meta, core::storage_t storage, size_t offset = 0);

public:
    static tensor_t create(
        const std::vector<size_t> &shape,
        llaisysDataType_t dtype,
        llaisysDeviceType_t device_type = LLAISYS_DEVICE_CPU,
        int device = 0);
    ~Tensor() = default;
    // Info
    std::byte *data();
    const std::byte *data() const;
    size_t ndim() const;
    const std::vector<size_t> &shape() const;
    const std::vector<ptrdiff_t> &strides() const;
    llaisysDataType_t dtype() const;
    llaisysDeviceType_t deviceType() const;
    int deviceId() const;
    size_t numel() const;
    size_t elementSize() const;

    std::string info() const;
    void debug() const;

    bool isContiguous() const;

    // Meta Transform
    tensor_t permute(const std::vector<size_t> &order) const;
    tensor_t slice(size_t dim, size_t start, size_t end) const;
    tensor_t view(const std::vector<size_t> &shape) const;

    // Load data from host memory
    void load(const void *src);

    // Challenging features
    tensor_t contiguous() const;
    tensor_t reshape(const std::vector<size_t> &shape) const;
    tensor_t to(llaisysDeviceType_t device_type, int device = -1) const;
};

} // namespace llaisys
