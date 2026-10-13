"""Exercise the installed native wheel without framework dependencies."""

import ctypes

from safetensors import TensorSpec, deserialize, serialize

data = (ctypes.c_ubyte * 4)(1, 2, 3, 4)
tensor = TensorSpec(dtype="uint8", shape=[4], data_ptr=ctypes.addressof(data), data_len=len(data))
assert dict(deserialize(serialize({"x": tensor}))) == {
    "x": {"dtype": "U8", "shape": [4], "data": bytes(data)}
}
