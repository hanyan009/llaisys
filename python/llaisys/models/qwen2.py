from typing import Sequence
from ..libllaisys import LIB_LLAISYS
from ..libllaisys import DeviceType, DataType
from ..libllaisys.qwen2 import LlaisysQwen2Meta
from ..tensor import Tensor
from pathlib import Path
import json
from ctypes import c_int, c_int64, c_void_p


class Qwen2:

    def __init__(self, model_path, device: DeviceType = DeviceType.CPU):
        model_path = Path(model_path)
        
        print("[Qwen2.__init__] Step 1: Loading config...")
        # Load config
        with open(model_path / "config.json") as f:
            config = json.load(f)
        
        print("[Qwen2.__init__] Step 2: Creating meta...")
        # Create meta
        meta = LlaisysQwen2Meta()
        meta.dtype = DataType.BF16
        meta.nlayer = config["num_hidden_layers"]
        meta.hs = config["hidden_size"]
        meta.nh = config["num_attention_heads"]
        meta.nkvh = config["num_key_value_heads"]
        meta.dh = config["hidden_size"] // config["num_attention_heads"]
        meta.di = config["intermediate_size"]
        meta.maxseq = config.get("max_position_embeddings", 32768)
        meta.voc = config["vocab_size"]
        meta.epsilon = config["rms_norm_eps"]
        meta.theta = config["rope_theta"]
        meta.end_token = config["eos_token_id"]
        
        print(f"[Qwen2.__init__] Meta: nlayer={meta.nlayer}, hs={meta.hs}, voc={meta.voc}")
        
        print("[Qwen2.__init__] Step 3: Creating C++ model...")
        # Create model
        from ctypes import byref
        device_id = c_int(0)
        self._model = LIB_LLAISYS.llaisysQwen2ModelCreate(meta, device, byref(device_id), 1)
        print(f"[Qwen2.__init__] Model created: {self._model}")
        
        print("[Qwen2.__init__] Step 4: Getting weights pointer...")
        self._weights_ptr = LIB_LLAISYS.llaisysQwen2ModelWeights(self._model)
        print(f"[Qwen2.__init__] Weights pointer: {self._weights_ptr}")
        
        print("[Qwen2.__init__] Step 5: Accessing weights contents...")
        self._weights = self._weights_ptr.contents
        self._nlayer = meta.nlayer
        self._device = device
        self._end_token = meta.end_token
        print("[Qwen2.__init__] Weights contents accessed successfully")
        
        # Load weights
        import mmap
        import struct
        
        print("Loading model weights...")
        total_files = len(list(model_path.glob("*.safetensors")))
        
        for file_idx, file in enumerate(sorted(model_path.glob("*.safetensors")), 1):
            print(f"  [{file_idx}/{total_files}] Loading {file.name}...")
            
            # Use mmap for faster loading
            with open(file, 'rb') as f:
                # Read header length
                header_len_bytes = f.read(8)
                header_len = struct.unpack('<Q', header_len_bytes)[0]
                
                # Map the entire file
                mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
                
                # Read header
                header_bytes = mm[8:8+header_len]
                header = json.loads(header_bytes.decode('utf-8'))
                
                # Load each tensor
                tensor_count = len([k for k in header.keys() if k != '__metadata__'])
                for idx, (name, info) in enumerate(header.items()):
                    if name == '__metadata__':
                        continue
                    
                    # Get tensor info
                    dtype = info['dtype']
                    shape = info['shape']
                    data_offsets = info['data_offsets']
                    
                    # Get data view directly from mmap (zero-copy)
                    start = 8 + header_len + data_offsets[0]
                    end = 8 + header_len + data_offsets[1]
                    data_bytes = mm[start:end]
                    
                    # Convert to numpy array (still a view, no copy)
                    array = self._bytes_to_numpy_view(data_bytes, dtype, shape)
                    self._load_weight(name, array)
                
                mm.close()
        
        print("Model loaded successfully!")

    def _bytes_to_numpy_view(self, data_bytes, dtype_str, shape):
        """Zero-copy conversion from mmap bytes to numpy array"""
        import numpy as np
        
        # Map safetensors dtype to numpy dtype
        dtype_map = {
            'F32': np.float32,
            'F64': np.float64,
            'I32': np.int32,
            'I64': np.int64,
            'U8': np.uint8,
            'I8': np.int8,
            'F16': np.float16,
            'BF16': np.uint16,  # bfloat16 stored as uint16
        }
        
        np_dtype = dtype_map.get(dtype_str, np.float32)
        
        # Create numpy array as a view (zero-copy)
        array = np.frombuffer(data_bytes, dtype=np_dtype)
        
        # Make a copy since the mmap will be closed
        array = np.copy(array)
        
        # Reshape
        if len(shape) > 0:
            array = array.reshape(shape)
        
        return array
    
    def _bytes_to_numpy(self, data_bytes, dtype_str, shape):
        import numpy as np
        
        # Map safetensors dtype to numpy dtype
        dtype_map = {
            'F32': np.float32,
            'F64': np.float64,
            'I32': np.int32,
            'I64': np.int64,
            'U8': np.uint8,
            'I8': np.int8,
            'F16': np.float16,
            'BF16': np.uint16,  # bfloat16 stored as uint16
        }
        
        np_dtype = dtype_map.get(dtype_str, np.float32)
        
        # Create numpy array from bytes
        array = np.frombuffer(data_bytes, dtype=np_dtype)
        
        # Reshape
        if len(shape) > 0:
            array = array.reshape(shape)
        
        return array

    def _load_weight(self, name: str, data):
        import numpy as np
        
        # Ensure it's a numpy array
        if not isinstance(data, np.ndarray):
            data = np.array(data)
        
        # Debug: print shape
        print(f"    Loading {name}: shape={data.shape}, dtype={data.dtype}")
        
        # Create tensor from numpy
        tensor = Tensor.from_numpy(data)
        
        # Get destination tensor handle and wrap it
        if name == "model.embed_tokens.weight":
            # Convert int to c_void_p if needed
            if isinstance(self._weights.in_embed, int):
                tensor_ptr = c_void_p(self._weights.in_embed)
            else:
                tensor_ptr = self._weights.in_embed
            
            print(f"      Tensor pointer: {tensor_ptr} (type: {type(tensor_ptr)})")
            print(f"      Tensor pointer value: {tensor_ptr.value if hasattr(tensor_ptr, 'value') else tensor_ptr}")
            
            dst = Tensor(tensor=tensor_ptr, need_destruct=False)
            print(f"      Destination tensor created")
            print(f"      Destination._tensor: {dst._tensor}")
            print(f"      Calling ndim...")
            ndim = dst.ndim()
            print(f"      Destination ndim: {ndim}")
            if ndim > 0:
                print(f"      Destination shape: {dst.shape()}")
            tensor.copy_to(dst)
        elif name == "lm_head.weight":
            tensor_ptr = c_void_p(self._weights.out_embed) if isinstance(self._weights.out_embed, int) else self._weights.out_embed
            dst = Tensor(tensor=tensor_ptr, need_destruct=False)
            tensor.copy_to(dst)
        elif name == "model.norm.weight":
            tensor_ptr = c_void_p(self._weights.out_norm_w) if isinstance(self._weights.out_norm_w, int) else self._weights.out_norm_w
            dst = Tensor(tensor=tensor_ptr, need_destruct=False)
            tensor.copy_to(dst)
        else:
            # Parse layer weights
            for i in range(self._nlayer):
                layer_prefix = f"model.layers.{i}."
                if name.startswith(layer_prefix):
                    suffix = name[len(layer_prefix):]
                    dst = None
                    tensor_ptr = None
                    if suffix == "input_layernorm.weight":
                        tensor_ptr = self._weights.attn_norm_w[i]
                    elif suffix == "self_attn.q_proj.weight":
                        tensor_ptr = self._weights.attn_q_w[i]
                    elif suffix == "self_attn.q_proj.bias":
                        tensor_ptr = self._weights.attn_q_b[i]
                    elif suffix == "self_attn.k_proj.weight":
                        tensor_ptr = self._weights.attn_k_w[i]
                    elif suffix == "self_attn.k_proj.bias":
                        tensor_ptr = self._weights.attn_k_b[i]
                    elif suffix == "self_attn.v_proj.weight":
                        tensor_ptr = self._weights.attn_v_w[i]
                    elif suffix == "self_attn.v_proj.bias":
                        tensor_ptr = self._weights.attn_v_b[i]
                    elif suffix == "self_attn.o_proj.weight":
                        tensor_ptr = self._weights.attn_o_w[i]
                    elif suffix == "post_attention_layernorm.weight":
                        tensor_ptr = self._weights.mlp_norm_w[i]
                    elif suffix == "mlp.gate_proj.weight":
                        tensor_ptr = self._weights.mlp_gate_w[i]
                    elif suffix == "mlp.up_proj.weight":
                        tensor_ptr = self._weights.mlp_up_w[i]
                    elif suffix == "mlp.down_proj.weight":
                        tensor_ptr = self._weights.mlp_down_w[i]
                    
                    if tensor_ptr is not None:
                        # Convert int to c_void_p if needed
                        if isinstance(tensor_ptr, int):
                            tensor_ptr = c_void_p(tensor_ptr)
                        dst = Tensor(tensor=tensor_ptr, need_destruct=False)
                    
                    if dst is not None:
                        try:
                            tensor.copy_to(dst)
                        except Exception as e:
                            print(f"Error copying {name}: {e}")
                            print(f"  Source shape: {data.shape}, dtype: {data.dtype}")
                            print(f"  Dest shape: {dst.shape()}")
                            raise
                    break

    def generate(
        self,
        inputs: Sequence[int],
        max_new_tokens: int = None,
        top_k: int = 1,
        top_p: float = 0.8,
        temperature: float = 0.8,
    ):
        output_ids = list(inputs)
        
        # First token generation with full prompt
        token_array = (c_int64 * len(output_ids))(*output_ids)
        next_token = LIB_LLAISYS.llaisysQwen2ModelInfer(self._model, token_array, len(output_ids), temperature, top_p, top_k)
        output_ids.append(next_token)
        
        # Generate remaining tokens one by one
        steps = 0
        while (max_new_tokens is None or steps < max_new_tokens - 1) and next_token != self._end_token:
            token_array = (c_int64 * 1)(next_token)
            next_token = LIB_LLAISYS.llaisysQwen2ModelInfer(self._model, token_array, 1, temperature, top_p, top_k)
            output_ids.append(next_token)
            steps += 1
        
        return output_ids

    def generate_stream(
        self,
        inputs: Sequence[int],
        max_new_tokens: int = None,
        top_k: int = 1,
        top_p: float = 0.8,
        temperature: float = 0.8,
    ):
        # First token generation with full prompt
        output_ids = list(inputs)
        token_array = (c_int64 * len(output_ids))(*output_ids)
        next_token = LIB_LLAISYS.llaisysQwen2ModelInfer(self._model, token_array, len(output_ids), temperature, top_p, top_k)
        yield next_token
        
        # Generate remaining tokens one by one
        steps = 0
        while (max_new_tokens is None or steps < max_new_tokens - 1) and next_token != self._end_token:
            token_array = (c_int64 * 1)(next_token)
            next_token = LIB_LLAISYS.llaisysQwen2ModelInfer(self._model, token_array, 1, temperature, top_p, top_k)
            yield next_token
            steps += 1
    
    def __del__(self):
        if hasattr(self, '_model'):
            LIB_LLAISYS.llaisysQwen2ModelDestroy(self._model)
