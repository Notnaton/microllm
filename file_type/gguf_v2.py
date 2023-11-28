import struct
from enum import IntEnum
from pydantic import BaseModel
from typing import List, Optional, Union

class ggml_type(IntEnum):
    GGML_TYPE_F32  = 0
    GGML_TYPE_F16  = 1
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
    GGML_TYPE_I8 = 16
    GGML_TYPE_I16 = 17
    GGML_TYPE_I32 = 18
    GGML_TYPE_COUNT = 19


class gguf_metadata_value_type(IntEnum):
    UINT8 = 0
    INT8 = 1
    UINT16 = 2
    INT16 = 3
    UINT32 = 4
    INT32 = 5
    FLOAT32 = 6
    BOOL = 7
    # The value is a UTF-8 non-null-terminated string, with length prepended.
    STRING = 8
    # The value is an array of other values, with the length and type prepended.
    #/
    # Arrays can be nested, and the length of the array is the number of elements in the array, not the number of bytes.
    ARRAY = 9
    UINT64 = 10
    INT64 = 11
    FLOAT64 = 12

class gguf_string_t(BaseModel):
    len: int
    string: str

class gguf_metadata_value_t(BaseModel):
    type: gguf_metadata_value_type
    len: int
    data: Union[List[type], type]

class gguf_metadata_kv_t(BaseModel):
    key: gguf_string_t
    value_type: gguf_metadata_value_type
    value: gguf_metadata_value_t 

class gguf_header_t(BaseModel):
    magic: str
    version: int
    tensor_count: int
    metadata_kv_count: int
    metadata_kv: List[gguf_metadata_kv_t]

class gguf_tensor_info_t(BaseModel):
    name: gguf_string_t
    n_dimensions: int
    dimensions: List[int]
    type: ggml_type 
    offset: int

class gguf_file(BaseModel):
    header: gguf_header_t
    tensor_infos: List[gguf_tensor_info_t]
    padding: List[bytes]
    tensor_data: List[int]

def read_string(file) -> gguf_string_t:
    read_str = gguf_string_t
    read_str.len = int.from_bytes(file.read(8), byteorder='little')
    read_str.string = file.read(read_str.len).decode('utf-8')
    return read_str

def read_metadata_value_t(file, value_type) -> gguf_metadata_value_t:
    gmvt = gguf_metadata_value_t
    gmvt.type = gguf_metadata_value_type(value_type)

    buffer = []
    match gmvt.type:
        case gguf_metadata_value_type.INT8:
            buffer.append(struct.unpack('b', file.read(1))[0])
        case gguf_metadata_value_type.UINT8:
            buffer.append(struct.unpack('B', file.read(1))[0])
        case gguf_metadata_value_type.BOOL:
            buffer.append(struct.unpack('B', file.read(1))[0])
        case gguf_metadata_value_type.INT16:
            buffer.append(struct.unpack('<h', file.read(2))[0])
        case gguf_metadata_value_type.UINT16:
            buffer.append(struct.unpack('<H', file.read(2))[0])
        case gguf_metadata_value_type.INT32:
            buffer.append(struct.unpack('<i', file.read(4))[0])
        case gguf_metadata_value_type.UINT32:
            buffer.append(struct.unpack('<I', file.read(4))[0])
        case gguf_metadata_value_type.FLOAT32:
            buffer.append(struct.unpack('<f', file.read(4))[0])
        case gguf_metadata_value_type.INT64:
            buffer.append(struct.unpack('<q', file.read(8))[0])
        case gguf_metadata_value_type.UINT64:
            buffer.append(struct.unpack('<Q', file.read(8))[0])
        case gguf_metadata_value_type.FLOAT64:
            buffer.append(struct.unpack('<d', file.read(8))[0])
        case gguf_metadata_value_type.STRING:
            buffer.append(read_string(file).string)
        case gguf_metadata_value_type.ARRAY:
            array_type = int.from_bytes(file.read(4), byteorder='little')
            array_length = int.from_bytes(file.read(8), byteorder='little')
            buffer = [read_metadata_value_t(file, array_type).data for _ in range(array_length)]
    gmvt.data = buffer
    return gmvt

def read_metadata_kv(file) -> gguf_metadata_kv_t:
    metadata = gguf_metadata_kv_t
    metadata.key =          read_string(file).string
    metadata.value_type =   gguf_metadata_value_type(int.from_bytes(file.read(4), byteorder='little'))
    metadata.value = read_metadata_value_t(file, metadata.value_type).data
    return metadata

def read_header(file) -> gguf_header_t:
    gguf_header = gguf_header_t
    gguf_header.magic =             file.read(4).decode('utf-8')
    gguf_header.version =           int.from_bytes(file.read(4), byteorder='little')
    gguf_header.tensor_count =      int.from_bytes(file.read(8), byteorder='little')
    gguf_header.metadata_kv_count = int.from_bytes(file.read(8), byteorder='little')
    buffer=[]
    for _ in range(gguf_header.metadata_kv_count):
        buffer.append(read_metadata_kv(file))
    gguf_header.metadata_kv = buffer

    return gguf_header

def read_tensor_infos(file) -> gguf_tensor_info_t:
    t_info = gguf_tensor_info_t
    t_info.name = read_string(file).string
    t_info.n_dimensions = int.from_bytes(file.read(4), byteorder='little')
    t_info.dimensions = [int.from_bytes(file.read(8), byteorder='little') for _ in range(t_info.n_dimensions)]
    t_info.type = ggml_type(int.from_bytes(file.read(4), byteorder='little'))
    t_info.offset = int.from_bytes(file.read(8), byteorder='little')
    return t_info

def align_offset(offset, ALIGNMENT) -> int:
    return offset + (ALIGNMENT - (offset % ALIGNMENT)) % ALIGNMENT

def read_gguf(file) -> gguf_file:
    g_file = gguf_file
    with open(file, mode="rb") as f:
        g_file.header = read_header(f)
        g_file.tensor_infos = [read_tensor_infos(f) for _ in range(g_file.header.tensor_count)]
        #g_file.padding = 
        #g_file.tensor_data =
    return g_file

if __name__ == "__main__":
    #gguf_data = read_gguf("E:\LLM\models\TheBloke\zephyr-7B-beta-GGUF\zephyr-7b-beta.Q4_K_S.gguf")
    gguf_data = read_gguf("E:\LLM\models\TheBloke\Mistral-7B-Instruct-v0.1-GGUF\mistral-7b-instruct-v0.1.Q4_0.gguf")
    print(gguf_data.tensor_infos)