from typing import Sequence, Tuple

from .libllaisys import (
    LIB_LLAISYS,
    llaisysTensor_t,
    llaisysDeviceType_t,
    DeviceType,
    llaisysDataType_t,
    DataType,
)
from ctypes import c_size_t, c_int, c_ssize_t, c_void_p


class Tensor:
    def __init__(
        self,
        shape: Sequence[int] = None,
        dtype: DataType = DataType.F32,
        device: DeviceType = DeviceType.CPU,
        device_id: int = 0,
        tensor: llaisysTensor_t = None,
        need_destruct: bool = True,
    ):
        if tensor:
            self._tensor = tensor
            self._need_destruct = need_destruct
        else:
            _ndim = 0 if shape is None else len(shape)
            _shape = None if shape is None else (c_size_t * len(shape))(*shape)
            self._tensor: llaisysTensor_t = LIB_LLAISYS.tensorCreate(
                _shape,
                c_size_t(_ndim),
                llaisysDataType_t(dtype),
                llaisysDeviceType_t(device),
                c_int(device_id),
            )
            self._need_destruct = True

    def __del__(self):
        if hasattr(self, "_tensor") and self._tensor is not None and hasattr(self, "_need_destruct") and self._need_destruct:
            LIB_LLAISYS.tensorDestroy(self._tensor)
            self._tensor = None

    def shape(self) -> Tuple[int]:
        buf = (c_size_t * self.ndim())()
        LIB_LLAISYS.tensorGetShape(self._tensor, buf)
        return tuple(buf[i] for i in range(self.ndim()))

    def strides(self) -> Tuple[int]:
        buf = (c_ssize_t * self.ndim())()
        LIB_LLAISYS.tensorGetStrides(self._tensor, buf)
        return tuple(buf[i] for i in range(self.ndim()))

    def ndim(self) -> int:
        return int(LIB_LLAISYS.tensorGetNdim(self._tensor))

    def dtype(self) -> DataType:
        return DataType(LIB_LLAISYS.tensorGetDataType(self._tensor))

    def device_type(self) -> DeviceType:
        return DeviceType(LIB_LLAISYS.tensorGetDeviceType(self._tensor))

    def device_id(self) -> int:
        return int(LIB_LLAISYS.tensorGetDeviceId(self._tensor))

    def data_ptr(self) -> c_void_p:
        return LIB_LLAISYS.tensorGetData(self._tensor)

    def lib_tensor(self) -> llaisysTensor_t:
        return self._tensor

    def debug(self):
        LIB_LLAISYS.tensorDebug(self._tensor)

    def __repr__(self):
        return f"<Tensor shape={self.shape}, dtype={self.dtype}, device={self.device_type}:{self.device_id}>"

    def load(self, data: c_void_p):
        LIB_LLAISYS.tensorLoad(self._tensor, data)

    def is_contiguous(self) -> bool:
        return bool(LIB_LLAISYS.tensorIsContiguous(self._tensor))

    def view(self, *shape: int) -> llaisysTensor_t:
        _shape = (c_size_t * len(shape))(*shape)
        return Tensor(
            tensor=LIB_LLAISYS.tensorView(self._tensor, _shape, c_size_t(len(shape)))
        )

    def permute(self, *perm: int) -> llaisysTensor_t:
        assert len(perm) == self.ndim()
        _perm = (c_size_t * len(perm))(*perm)
        return Tensor(tensor=LIB_LLAISYS.tensorPermute(self._tensor, _perm))

    def slice(self, dim: int, start: int, end: int):
        return Tensor(
            tensor=LIB_LLAISYS.tensorSlice(
                self._tensor, c_size_t(dim), c_size_t(start), c_size_t(end)
            )
        )

    @staticmethod
    def from_numpy(array):
        import numpy as np
        
        # Ensure it's a numpy array
        if not isinstance(array, np.ndarray):
            try:
                array = np.array(array)
            except:
                # If conversion fails, try to get raw data
                array = np.asarray(array)
        
        # Ensure contiguous
        if not array.flags['C_CONTIGUOUS']:
            array = np.ascontiguousarray(array)
        
        dtype_map = {
            np.float32: DataType.F32,
            np.float64: DataType.F64,
            np.int32: DataType.I32,
            np.int64: DataType.I64,
            np.uint8: DataType.U8,
            np.int8: DataType.I8,
            np.float16: DataType.F16,
            np.uint16: DataType.BF16,  # bfloat16 is stored as uint16
        }
        
        # Handle bfloat16 - check both dtype name and actual type
        dtype_str = str(array.dtype)
        if 'bfloat16' in dtype_str or array.dtype == np.uint16:
            dtype = DataType.BF16
        else:
            dtype = dtype_map.get(array.dtype.type, DataType.F32)
        
        tensor = Tensor(shape=array.shape, dtype=dtype, device=DeviceType.CPU)
        tensor.load(array.ctypes.data_as(c_void_p))
        return tensor
    
    def copy_to(self, dst):
        src_shape = self.shape()
        dst_shape = dst.shape()
        assert src_shape == dst_shape, f"Shape mismatch: {src_shape} vs {dst_shape}"
        
        # Use load API which handles device transfer
        dst.load(self.data_ptr())
