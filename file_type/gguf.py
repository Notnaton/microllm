import struct
import numpy as np
from enum import IntEnum
from pydantic import BaseModel
from typing import List, Optional, Union

ALIGNMENT = 32

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

class gguf():
    def __init__(self, filename, byte_order='little'):
        self.file = open(filename, 'rb')
        self.byte_order = byte_order
        self.read_header()
        self.read_tensor_info()
        #for key in self.metadata_key.keys():
        #    print(self.metadata_key[key], key)
        #    x = input()
        print(self.header)
        for key in self.metadata_key.keys():
            if "tokenizer" not in key:
                print(f"{key} {self.metadata_key[key]}")
        #for tensor_info in self.tensor_infos:
        #    print(tensor_info)

    
    def read_header(self):
        self.header = {
            "magic":            str(self.file.read(4)),
            "version":          int.from_bytes(self.file.read(4), self.byte_order),
            "tensor_count":     int.from_bytes(self.file.read(8), self.byte_order),
            "metadata_kv_count":int.from_bytes(self.file.read(8), self.byte_order)
        }
        self.metadata_key = {}
        for _ in range(self.header["metadata_kv_count"]):
            str_len = int.from_bytes(self.file.read(8), self.byte_order)
            string = self.file.read(str_len).decode('utf-8')
            value_type_n = int.from_bytes(self.file.read(4), self.byte_order)
            value_type = gguf_metadata_value_type(value_type_n)
            #print(f"{str_len} {string} {value_type_n} {value_type}")

            self.metadata_key[string] = self.read_value(value_type)

    def read_value_type(self, value):
        value = gguf_metadata_value_type(value)
        match value:
            case gguf_metadata_value_type.UINT8:
                return 1
            case gguf_metadata_value_type.INT8:
                return 1
            case gguf_metadata_value_type.UINT16:
                return 2
            case gguf_metadata_value_type.INT16:
                return 2
            case gguf_metadata_value_type.UINT32:
                return 4
            case gguf_metadata_value_type.INT32:
                return 4
            case gguf_metadata_value_type.FLOAT32:
                return 4
            case gguf_metadata_value_type.BOOL:
                return 1
            case gguf_metadata_value_type.UINT64:
                return 8
            case gguf_metadata_value_type.INT64:
                return 8
            case gguf_metadata_value_type.FLOAT64:
                return 8
            case gguf_metadata_value_type.STRING:
                # For STRING, read its length and then set the file pointer back
                current_pos = self.file.tell()
                length = int.from_bytes(self.file.read(8), self.byte_order)
                self.file.seek(current_pos) # Return to previous position
                return 8 + length
            case gguf_metadata_value_type.ARRAY:
                # For ARRAY, return size for type and length fields
                return 4 + 8
            
    def read_value(self, value_type):
        value_type = gguf_metadata_value_type(value_type)
        match value_type:
            case gguf_metadata_value_type.UINT8:
                return int.from_bytes(self.file.read(1), self.byte_order)
            case gguf_metadata_value_type.INT8:
                return struct.unpack("b", self.file.read(1))[0]
            case gguf_metadata_value_type.UINT16:
                return int.from_bytes(self.file.read(2), self.byte_order)
            case gguf_metadata_value_type.INT16:
                return struct.unpack("h", self.file.read(2))[0]
            case gguf_metadata_value_type.UINT32:
                return int.from_bytes(self.file.read(4), self.byte_order)
            case gguf_metadata_value_type.INT32:
                return struct.unpack("i", self.file.read(4))[0]
            case gguf_metadata_value_type.FLOAT32:
                return struct.unpack("f", self.file.read(4))[0]
            case gguf_metadata_value_type.BOOL:
                # Assuming 0 is False and 1 is True
                return bool(self.file.read(1)[0])
            case gguf_metadata_value_type.STRING:
                length = int.from_bytes(self.file.read(8), self.byte_order)
                return self.file.read(length).decode('utf-8')
            case gguf_metadata_value_type.ARRAY:
                array_type = gguf_metadata_value_type(int.from_bytes(self.file.read(4), self.byte_order))
                array_length = int.from_bytes(self.file.read(8), self.byte_order)
                return [self.read_value(array_type) for _ in range(array_length)]
            case gguf_metadata_value_type.UINT64:
                return int.from_bytes(self.file.read(8), self.byte_order)
            case gguf_metadata_value_type.INT64:
                return struct.unpack("q", self.file.read(8))[0]
            case gguf_metadata_value_type.FLOAT64:
                return struct.unpack("d", self.file.read(8))[0]

    def read_tensor_info(self):
        self.tensor_infos = []
        for _ in range(self.header["tensor_count"]):
            tensor_info = {}

            # Read tensor name
            str_len = int.from_bytes(self.file.read(8), self.byte_order)
            tensor_info["name"] = self.file.read(str_len).decode('utf-8')

            # Read number of dimensions
            tensor_info["n_dimensions"] = int.from_bytes(self.file.read(4), self.byte_order)

            # Read dimensions
            tensor_info["dimensions"] = [int.from_bytes(self.file.read(8), self.byte_order) 
                                        for _ in range(tensor_info["n_dimensions"])]

            # Read type of tensor
            tensor_info["type"] = ggml_type(int.from_bytes(self.file.read(4), self.byte_order))

            # Read offset
            tensor_info["offset"] = int.from_bytes(self.file.read(8), self.byte_order)

            self.tensor_infos.append(tensor_info)

        # Calculate padding (if needed)
        current_position = self.file.tell()
        padding_size = ALIGNMENT - (current_position % ALIGNMENT)
        if padding_size < ALIGNMENT:
            self.file.read(padding_size)  # Skip padding bytes

    def get_numpy_dtype(self, tensor_type):
        # Map the ggml_type to corresponding numpy dtype
        type_map = {
            ggml_type.GGML_TYPE_F32: np.float32,
            ggml_type.GGML_TYPE_I8: np.int8,
            # ... [add other type mappings]
        }
        return type_map[tensor_type]

    def close(self):
        self.file.close()
    
    def __del__(self):
        #print(self.metadata_key)
        self.close()

if __name__ == "__main__":
    gguf("E:\LLM\models\TheBloke\Mistral-7B-Instruct-v0.1-GGUF\mistral-7b-instruct-v0.1.Q4_0.gguf")