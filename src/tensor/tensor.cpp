#include "tensor.hpp"

#include "../utils.hpp"

#include <cstring>
#include <functional>
#include <numeric>
#include <sstream>

namespace llaisys {

Tensor::Tensor(TensorMeta meta, core::storage_t storage, size_t offset)
    : _meta(std::move(meta)), _storage(std::move(storage)), _offset(offset) {} // std::move 移交指针所有权

tensor_t Tensor::create(const std::vector<size_t> &shape,
                        llaisysDataType_t dtype,
                        llaisysDeviceType_t device_type,
                        int device) {
    size_t ndim_ = shape.size();
    std::vector<ptrdiff_t> strides(ndim_);
    size_t stride = 1;
    for (size_t i = 1; i <= ndim_; i++) {
        strides[ndim_ - i] = stride;
        stride *= shape[ndim_ - i];
    }
    TensorMeta meta{dtype, shape, strides};
    size_t total_elems = stride;
    size_t dtype_size = utils::dsize(dtype);

    if (device_type == LLAISYS_DEVICE_CPU && core::context().runtime().deviceType() != LLAISYS_DEVICE_CPU) {
        auto storage = core::context().runtime().allocateHostStorage(total_elems * dtype_size);
        return std::shared_ptr<Tensor>(new Tensor(meta, storage));
    } else {
        core::context().setDevice(device_type, device);
        auto storage = core::context().runtime().allocateDeviceStorage(total_elems * dtype_size);
        return std::shared_ptr<Tensor>(new Tensor(meta, storage));
    }
}

std::byte *Tensor::data() {
    return _storage->memory() + _offset;
}

const std::byte *Tensor::data() const {
    return _storage->memory() + _offset;
}

size_t Tensor::ndim() const {
    return _meta.shape.size();
}

const std::vector<size_t> &Tensor::shape() const {
    return _meta.shape;
}

const std::vector<ptrdiff_t> &Tensor::strides() const {
    return _meta.strides;
}

llaisysDataType_t Tensor::dtype() const {
    return _meta.dtype;
}

llaisysDeviceType_t Tensor::deviceType() const { // 从storage中拿到devicetype
    return _storage->deviceType();
}

int Tensor::deviceId() const {
    return _storage->deviceId();
}

size_t Tensor::numel() const {
    return std::accumulate(_meta.shape.begin(), _meta.shape.end(), size_t(1), std::multiplies<size_t>());
}

size_t Tensor::elementSize() const {
    return utils::dsize(_meta.dtype);
}

std::string Tensor::info() const {
    std::stringstream ss;

    ss << "Tensor: "
       << "shape[ ";
    for (auto s : this->shape()) {
        ss << s << " ";
    }
    ss << "] strides[ ";
    for (auto s : this->strides()) {
        ss << s << " ";
    }
    ss << "] dtype=" << this->dtype();

    return ss.str();
}

template <typename T>
void print_data(const T *data, const std::vector<size_t> &shape, const std::vector<ptrdiff_t> &strides, size_t dim) {
    if (dim == shape.size() - 1) {
        for (size_t i = 0; i < shape[dim]; i++) {
            if constexpr (std::is_same_v<T, bf16_t> || std::is_same_v<T, fp16_t>) {
                std::cout << utils::cast<float>(data[i * strides[dim]]) << " ";
            } else {
                std::cout << data[i * strides[dim]] << " ";
            }
        }
        std::cout << std::endl;
    } else if (dim < shape.size() - 1) {
        for (size_t i = 0; i < shape[dim]; i++) {
            print_data(data + i * strides[dim], shape, strides, dim + 1);
        }
    }
}

void debug_print(const std::byte *data, const std::vector<size_t> &shape, const std::vector<ptrdiff_t> &strides, llaisysDataType_t dtype) {
    switch (dtype) {
    case LLAISYS_DTYPE_BYTE:
        return print_data(reinterpret_cast<const char *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_BOOL:
        return print_data(reinterpret_cast<const bool *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I8:
        return print_data(reinterpret_cast<const int8_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I16:
        return print_data(reinterpret_cast<const int16_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I32:
        return print_data(reinterpret_cast<const int32_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I64:
        return print_data(reinterpret_cast<const int64_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U8:
        return print_data(reinterpret_cast<const uint8_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U16:
        return print_data(reinterpret_cast<const uint16_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U32:
        return print_data(reinterpret_cast<const uint32_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U64:
        return print_data(reinterpret_cast<const uint64_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_F16:
        return print_data(reinterpret_cast<const fp16_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_F32:
        return print_data(reinterpret_cast<const float *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_F64:
        return print_data(reinterpret_cast<const double *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_BF16:
        return print_data(reinterpret_cast<const bf16_t *>(data), shape, strides, 0);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

void Tensor::debug() const {
    core::context().setDevice(this->deviceType(), this->deviceId());
    core::context().runtime().api()->device_synchronize();
    std::cout << this->info() << std::endl;
    if (this->deviceType() == LLAISYS_DEVICE_CPU) {
        debug_print(this->data(), this->shape(), this->strides(), this->dtype());
    } else {
        auto tmp_tensor = create({this->_storage->size()}, this->dtype());
        core::context().runtime().api()->memcpy_sync(
            tmp_tensor->data(),
            this->data(),
            this->numel() * this->elementSize(),
            LLAISYS_MEMCPY_D2H);
        debug_print(tmp_tensor->data(), this->shape(), this->strides(), this->dtype());
    }
}

bool Tensor::isContiguous() const {
    // TO_BE_IMPLEMENTED();
    // stride 第一个维度非 1，绝对非连续
    if (this->_meta.strides[this->_meta.shape.size() - 1] != 1) return false;
    // 此外还需满足：当前Stride = 上一个Stride 乘以 上一个维度的大小。也就是刚刚好
    for (size_t i = 0; i < this->_meta.shape.size() - 1; i++) {
        if ((size_t)this->_meta.strides[i] != this->_meta.strides[i + 1] * this->_meta.shape[i + 1]) {
            return false;
        }
    }
    return true;
}

tensor_t Tensor::permute(const std::vector<size_t> &order) const {
    // TO_BE_IMPLEMENTED();
    auto new_meta = _meta;
    for(size_t i = 0; i < _meta.shape.size(); i++){
         new_meta.shape[i] = _meta.shape[order[i]];
         new_meta.strides[i] = _meta.strides[order[i]];
    }
    return std::shared_ptr<Tensor>(new Tensor(new_meta, _storage, _offset));
}

tensor_t Tensor::view(const std::vector<size_t> &shape) const {
    // 【说明】view 不修改原有变量，只是更改meta data。首先检查元素数量，与原始的张量是否为contiguous

    // 首先保证view所要求的元素个数相同
    size_t numel = std::accumulate(shape.begin(), shape.end(), size_t(1), std::multiplies<size_t>());
    if (numel != this->numel()) {
        ASSERT(false, "Tensor view size does not match");
    }

    // 检查：是否连续，因为计算view之前是要假设输入是contiguous的。
    if (!this->isContiguous()) { 
        ASSERT(false, "Tensor must be contiguous before view operation");
    }

    TensorMeta meta_new = _meta; // 创建新的meta data，便于修改

    meta_new.shape = shape; // shape 赋值为新的
    // stride 重新计算
    meta_new.strides.resize(shape.size()); // 调整strides大小与新shape一致
    meta_new.strides.back() = 1; // 最后一个维度的stride为1，最终当然是优先为contiguous的，
        
    // 从后往前计算stride
    for(int i = static_cast<int>(meta_new.strides.size()) - 2; i >= 0; i--){
        meta_new.strides[i] = meta_new.strides[i + 1] * meta_new.shape[i + 1];
    }

    return std::shared_ptr<Tensor>(new Tensor(meta_new, _storage, this->_offset));
}

tensor_t Tensor::slice(size_t dim, size_t start, size_t end) const {
    // slice 操作只改变 Offset 和 meta，不改变stride

    ASSERT(dim < _meta.shape.size(), "Dimension out of range");
    ASSERT(start <= end , "start must be less than or equal to end");
    ASSERT(end <= _meta.shape[dim] , "end is out of bounds for this dimension");

    auto meta_new = _meta;
    meta_new.shape[dim] = end - start;

    // offset 是以字节为单位的。所以变更offset的时候，要乘上this->elementSize()
    size_t new_offset = this->_offset + static_cast<size_t>(start * _meta.strides[dim]) * this->elementSize();
    
    return std::shared_ptr<Tensor>(new Tensor(meta_new, _storage, new_offset));
}

void Tensor::load(const void *src_) { // 【问】为什么load使用void &src?【答】void* 是通用类型，load时，只需关心起始地址和长度，不用关心具体类型
    // 设置设备上下文为当前张量所在的设备
    core::context().setDevice(this->deviceType(), this->deviceId());
    // context()能够实例化一个静态的（thread_local）的context对象，他有static性质和thread_local性质（每个线程独立一份此对象）
        // context 是线程单实例的。（thread_local）
    // 【问】为什么要context线程单实例？context是共享的执行环境，

    // 获取运行时 API
    auto api = core::context().runtime().api();

    // 计算数据大小
    size_t data_size = this->numel() * this->elementSize();

    // 【注】由于Tensor在内存中是线性存储的，所以有起始和结束地址，就能够完整拷贝。
    if (this->deviceType() == LLAISYS_DEVICE_CPU) {
        // CPU 设备：直接使用 memcpy 复制数据
        std::memcpy(this->data(), src_, data_size);
    } else {
        // 设备（如 GPU）：使用 memcpy_sync 从主机到设备拷贝
        api->memcpy_sync(
            this->data(),
            src_,
            data_size,
            LLAISYS_MEMCPY_H2D);
    }
}

tensor_t Tensor::contiguous() const {
    if (this->isContiguous()) {
        return std::shared_ptr<Tensor>(new Tensor(_meta, _storage, _offset));
    }

    // 创建新的连续 tensor
    auto new_tensor = Tensor::create(_meta.shape, _meta.dtype, this->deviceType(), this->deviceId());
    
    // 设置设备上下文
    core::context().setDevice(this->deviceType(), this->deviceId());
    auto api = core::context().runtime().api();
    
    size_t elem_size = this->elementSize();
    // size_t total_elems = this->numel();
    
    // 递归复制数据
    std::function<void(size_t, const std::byte*, std::byte*)> copy_recursive = 
        [&](size_t dim, const std::byte* src, std::byte* dst) {
        if (dim == _meta.shape.size()) {
            return;
        }
        
        if (dim == _meta.shape.size() - 1) {
            // 最内层维度：整块复制
            size_t copy_size = _meta.shape[dim] * elem_size;
            if (this->deviceType() == LLAISYS_DEVICE_CPU) {
                std::memcpy(dst, src, copy_size);
            } else {
                api->memcpy_sync(dst, src, copy_size, LLAISYS_MEMCPY_D2D);
            }
        } else {
            // 递归处理每个子维度
            for (size_t i = 0; i < _meta.shape[dim]; i++) {
                copy_recursive(dim + 1, 
                             src + i * _meta.strides[dim] * elem_size,
                             dst + i * new_tensor->strides()[dim] * elem_size);
            }
        }
    };
    
    copy_recursive(0, this->data(), new_tensor->data());
    
    return new_tensor;
}

tensor_t Tensor::reshape(const std::vector<size_t> &shape) const {
    TO_BE_IMPLEMENTED();
    return std::shared_ptr<Tensor>(new Tensor(_meta, _storage));
}

tensor_t Tensor::to(llaisysDeviceType_t device_type, int device) const {
    TO_BE_IMPLEMENTED();
    return std::shared_ptr<Tensor>(new Tensor(_meta, _storage));
}

} // namespace llaisys
