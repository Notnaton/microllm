import os
import struct
from enum import Enum, IntEnum
from typing import List, Optional, Union, Dict, NamedTuple

ALIGNMENT = 32

class ggml_type(Enum):
    F32 = 0
    F16 = 1
    Q4_0 = 2
    Q4_1 = 3
    Q5_0 = 6
    Q5_1 = 7
    Q8_0 = 8
    Q8_1 = 9
    Q2_K = 10
    Q3_K = 11
    Q4_K = 12
    Q5_K = 13
    Q6_K = 14
    Q8_K = 15
    I8 = 16
    I16 = 17
    I32 = 18

METADATA_VALUE_TYPE_MAP = {
    value: key for key, value in METADATA_VALUE_TYPE.__members__.items()
}

class gguf_metadata_kv(NamedTuple):
    key: str
    value_type: int
    value: Union[int, float, List['gguf_metadata_kv'], str]

def read_string(file) -> str:
    length = int.from_bytes(file.read(8), byteorder='little')
    return file.read(length).decode('utf-8')

def read_metadata_value(file, value_type) -> Union[int, float, List['gguf_metadata_kv'], str]:
    if 0 <= value_type < len(METADATA_VALUE_TYPE_MAP):
        value_map = METADATA_VALUE_TYPE_MAP[value_type]
        match value_map:
            case gguf_metadata_value_type.UINT8:
                return struct.unpack('B', file.read(1))[0]
            case gguf_metadata_value_type.INT8:
                return struct.unpack('b', file.read(1))[0]
            case gguf_metadata_value_type.BOOL:
                return struct.unpack('B', file.read(1))[0] != 0
            case gguf_metadata_value_type.UINT16:
                return struct.unpack('<H', file.read(2))[0]
            case gguf_metadata_value_type.INT16:
                return struct.unpack('<h', file.read(2))[0]
            case gguf_metadata_value_type.UINT32:
                return struct.unpack('<I', file.read(4))[0]
            case gguf_metadata_value_type.INT32:
                return struct.unpack('<i', file.read(4))[0]
            case gguf_metadata_value_type.FLOAT32:
                return struct.unpack('<f', file.read(4))[0]
            case gguf_metadata_value_type.INT64:
                return struct.unpack('<q', file.read(8))[0]
            case gguf_metadata_value_type.UINT64:
                return struct.unpack('<Q', file.read(8))[0]
            case gguf_metadata_value_type.FLOAT64:
                return struct.unpack('<d', file.read(8))[0]
            case gguf_metadata_value_type.STRING:
                length = int.from_bytes(file.read(8), byteorder='little')
                return read_string(file)
            case gguf_metadata_value_type.ARRAY:
                array_type = int.from_bytes(file.read(4), byteorder='little')
                length = int.from_bytes(file.read(8), byteorder='little')
                return [read_metadata_value(file, array_type) for _ in range(length)]
    else:
        raise ValueError(f"Invalid value type: {value_type}")

def read_metadata_kv_t(file) -> gguf_metadata_kv:
    key = read_string(file)
    value_type = int.from_bytes(file.read(4), byteorder='little')
    value = read_metadata_value(file, value_type)
    return gguf_metadata_kv(key, value_type, value)

def read_header(file) -> NamedTuple:
    magic = file.read(4).decode('utf-8')
    version = int.from_bytes(file.read(4), byteorder='little')
    tensor_count = int.from_bytes(file.read(8), byteorder='little')
    metadata_kv_count = int.from_bytes(file.read(8), byteorder='little')

    metadata_kvs = [read_metadata_kv_t(file) for _ in range(metadata_kv_count)]
    return magic, version, tensor_count, metadata_kv_count, metadata_kvs

def read_tensor_infos(file) -> List[NamedTuple]:
    def read_dimensions():
        return [int.from_bytes(file.read(8), byteorder='little') for _ in range(n_dimensions)]

    tensor_count = int.from_bytes(file.read(8), byteorder='little')
    tensor_infos = []
    for _ in range(tensor_count):
        name = read_string(file)
        n_dimensions = int.from_bytes(file.read(4), byteorder='little')
        dimensions = read_dimensions()
        type_value = int.from_bytes(file.read(4), byteorder='little')
        offset = int.from_bytes(file.read(8), byteorder='little')
        tensor_infos.append((name, n_dimensions, dimensions, type_value, offset))
    return tensor_infos

def align_offset(offset: int, alignment: int) -> int:
    return (offset + (alignment - 1)) & ~(alignment - 1)

def read_gguf(file_path: str) -> NamedTuple:
    with open(file_path, mode="rb") as file:
        magic, version, tensor_count, metadata_kv_count, metadata_kvs = read_header(file)
        tensor_infos = read_tensor_infos(file)

        padding = align_offset(file.tell(), ALIGNMENT)
        file.seek(0, os.SEEK_END)
        eof = file.tell()

        tensor_locations = []
        for info in tensor_infos:
            name, n_dimensions, dimensions, type_, offset = info
            end = align_offset(offset + sum(dimensions * getattr(ggml_type, type_).itemsize), ALIGNMENT)
            tensor_locations.append({"start": offset, "end": end})
            offset = end

        return magic, version, tensor_count, metadata_kv_count, metadata_kvs, tensor_infos, padding, tensor_locations