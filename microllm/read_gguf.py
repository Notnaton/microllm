"""
Read a file *.gguf and return file locations for the objects
"""

from typing import List, Union, Any
from dataclasses import dataclass
import struct
from enum import IntEnum
import numpy as np

ALIGNMENT = 32  # Assuming alignment is 32 if not specified

def align_offset(position, alignment=ALIGNMENT):
	return (position + alignment - 1) & ~(alignment - 1)

class GGMLType(IntEnum):
	GGML_TYPE_F32 = 0
	GGML_TYPE_F16 = 1
	GGML_TYPE_Q4_0 = 2
	GGML_TYPE_Q4_1 = 3
	GGML_TYPE_Q5_0 = 6
	GGML_TYPE_Q5_1 = 7
	GGML_TYPE_Q8_0 = 8
	GGML_TYPE_Q8_1 = 9
	GGML_TYPE_Q2_K = 10
	GGML_TYPE_Q3_K = 11
	GGML_TYPE_Q4_K = 12
	GGML_TYPE_Q5_K = 13
	GGML_TYPE_Q6_K = 14
	GGML_TYPE_Q8_K = 15
	GGML_TYPE_IQ2_XXS = 16
	GGML_TYPE_IQ2_XS = 17
	GGML_TYPE_IQ3_XXS = 18
	GGML_TYPE_IQ1_S = 19
	GGML_TYPE_IQ4_NL = 20
	GGML_TYPE_IQ3_S = 21
	GGML_TYPE_IQ2_S = 22
	GGML_TYPE_IQ4_XS = 23
	GGML_TYPE_I8 = 24
	GGML_TYPE_I16 = 25
	GGML_TYPE_I32 = 26
	GGML_TYPE_I64 = 27
	GGML_TYPE_F64 = 28
	GGML_TYPE_IQ1_M = 29
	GGML_TYPE_COUNT = 30

class GGUFMetadataValueType(IntEnum):
	GGUF_METADATA_VALUE_TYPE_UINT8 = 0
	GGUF_METADATA_VALUE_TYPE_INT8 = 1
	GGUF_METADATA_VALUE_TYPE_UINT16 = 2
	GGUF_METADATA_VALUE_TYPE_INT16 = 3
	GGUF_METADATA_VALUE_TYPE_UINT32 = 4
	GGUF_METADATA_VALUE_TYPE_INT32 = 5
	GGUF_METADATA_VALUE_TYPE_FLOAT32 = 6
	GGUF_METADATA_VALUE_TYPE_BOOL = 7
	GGUF_METADATA_VALUE_TYPE_STRING = 8
	GGUF_METADATA_VALUE_TYPE_ARRAY = 9
	GGUF_METADATA_VALUE_TYPE_UINT64 = 10
	GGUF_METADATA_VALUE_TYPE_INT64 = 11
	GGUF_METADATA_VALUE_TYPE_FLOAT64 = 12

@dataclass
class GGUFMetadataArray:
	def __init__(self, file):
		self.value_type: GGUFMetadataValueType = GGUFMetadataValueType(struct.unpack('<I', file.read(4))[0]); assert self.value_type in GGUFMetadataValueType
		self.len: int = struct.unpack('<Q', file.read(8))[0]
		self.array: List[Any] = [read_value(file, self.value_type) for _ in range(self.len)]

def gguf_string_t(file):
	len: int = struct.unpack('<Q', file.read(8))[0]
	return str(struct.unpack(f'{len}s', file.read(len))[0], encoding='utf-8')

def read_value(file, type: GGUFMetadataValueType):
	match type:
		case GGUFMetadataValueType.GGUF_METADATA_VALUE_TYPE_UINT8:
			value = struct.unpack('<B', file.read(1))[0]
		case GGUFMetadataValueType.GGUF_METADATA_VALUE_TYPE_INT8:
			value = struct.unpack('<b', file.read(1))[0]
		case GGUFMetadataValueType.GGUF_METADATA_VALUE_TYPE_UINT16:
			value = struct.unpack('<H', file.read(2))[0]
		case GGUFMetadataValueType.GGUF_METADATA_VALUE_TYPE_INT16:
			value = struct.unpack('<h', file.read(2))[0]
		case GGUFMetadataValueType.GGUF_METADATA_VALUE_TYPE_UINT32:
			value = struct.unpack('<I', file.read(4))[0]
		case GGUFMetadataValueType.GGUF_METADATA_VALUE_TYPE_INT32:
			value = struct.unpack('<i', file.read(4))[0]
		case GGUFMetadataValueType.GGUF_METADATA_VALUE_TYPE_FLOAT32:
			value = struct.unpack('<f', file.read(4))[0]
		case GGUFMetadataValueType.GGUF_METADATA_VALUE_TYPE_BOOL:
			value = struct.unpack('<?', file.read(1))[0]
		case GGUFMetadataValueType.GGUF_METADATA_VALUE_TYPE_STRING:
			value = gguf_string_t(file)
		case GGUFMetadataValueType.GGUF_METADATA_VALUE_TYPE_ARRAY:
			value = GGUFMetadataArray(file)
		case GGUFMetadataValueType.GGUF_METADATA_VALUE_TYPE_UINT64:
			value = struct.unpack('<Q', file.read(8))[0]
		case GGUFMetadataValueType.GGUF_METADATA_VALUE_TYPE_INT64:
			value = struct.unpack('<q', file.read(8))[0]
		case GGUFMetadataValueType.GGUF_METADATA_VALUE_TYPE_FLOAT64:
			value = struct.unpack('<d', file.read(8))[0]
	return value

@dataclass
class gguf_metadata_kv_t:
	def __init__(self, file):
		self.key: str = gguf_string_t(file)
		self.value_type: GGUFMetadataValueType = GGUFMetadataValueType(struct.unpack('<I', file.read(4))[0]); assert self.value_type in GGUFMetadataValueType
		self.value = read_value(file, self.value_type)

@dataclass
class gguf_header:
	def __init__(self, file):
		self.magic: str = struct.unpack('4s', file.read(4))[0]; assert self.magic == b'GGUF'
		self.version: int = struct.unpack('<I', file.read(4))[0]; assert self.version == 3
		self.tensor_count: int = struct.unpack('<Q', file.read(8))[0]; assert self.tensor_count > 0
		self.metadata_kv_count: int = struct.unpack('<Q', file.read(8))[0]; assert self.metadata_kv_count >= 0
		self.metadata_kv: List[gguf_metadata_kv_t] = [gguf_metadata_kv_t(file) for _ in range(self.metadata_kv_count)]

@dataclass
class gguf_tensor_info_t:
	def __init__(self, file):
		self.name: str = gguf_string_t(file); assert len(self.name) < 65
		self.n_dimensions: int = struct.unpack('<I', file.read(4))[0]
		self.dimensions: List[int] = [struct.unpack('<Q', file.read(8))[0] for _ in range(self.n_dimensions)]
		self.type: GGMLType = GGMLType(struct.unpack('<I', file.read(4))[0]); assert self.type in GGMLType
		self.offset: int = struct.unpack('<Q', file.read(8))[0]

@dataclass
class gguf_file:
	def __init__(self, file):
		self.header: gguf_header = gguf_header(file)
		self.tensor_info_t = [gguf_tensor_info_t(file) for _ in range(self.header.tensor_count)]
		self.padding = file.read(align_offset(file.tell()) - file.tell()); assert self.padding == b'\x00' * len(self.padding) #TODO: get alignment from header
		self.tensor_data_position = []
		self.tensor_data = {}
		for tensor in self.tensor_info_t:
			file.seek(tensor.offset)
			shape = tuple(tensor.dimensions)
			dtype = self.get_numpy_dtype(tensor.type)
			data = np.fromfile(file, dtype=dtype, count=np.prod(shape)).reshape(shape)
			self.tensor_data[tensor.name] = data

	def get_numpy_dtype(self, ggml_type):
		dtype_map = {
			GGMLType.GGML_TYPE_F32: np.float32,
			GGMLType.GGML_TYPE_F16: np.float16,
			GGMLType.GGML_TYPE_Q8_0: np.int8,
			# Add more mappings as needed
		}
		return dtype_map.get(ggml_type, np.float32)

	def to_dict(self):
		result = {
			'magic': self.header.magic,
			'version': self.header.version,
			'tensor_count': self.header.tensor_count,
			'metadata_kv_count': self.header.metadata_kv_count,
			'padding': self.padding,
			'tensor_data_position': self.tensor_data_position,
			'metadata': {},
			'tensor_info': {},
			'tensor_data': self.tensor_data
		}

		# Flatten metadata_kv
		for kv in self.header.metadata_kv:
			value = kv.value.array if isinstance(kv.value, GGUFMetadataArray) else kv.value
			result['metadata'][kv.key] = value
			result['metadata'][f'{kv.key}_type'] = kv.value_type

		# Flatten tensor_info_t
		for tensor in self.tensor_info_t:
			result['tensor_info'][tensor.name] = {
				'dimensions': tensor.dimensions,
				'type': tensor.type,
				'offset': tensor.offset
			}

		return result

if __name__ == '__main__':
	with open('/home/anton/.cache/lm-studio/models/lmstudio-community/Phi-3.5-mini-instruct-GGUF/Phi-3.5-mini-instruct-Q8_0.gguf', 'rb') as file:
		gguf = gguf_file(file).to_dict()
		for data in gguf:
			print(f"{data}: {gguf[data]}")	

