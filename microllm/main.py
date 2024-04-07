import numpy as np
from typing import List
from .file_type import gguf

#from devtools import pprint, debug

def main():
    data = gguf.read_gguf("/home/anton/Documents/models/laser-dolphin-mixtral-2x7b-dpo.Q5_K_M.gguf")
    #debug(data)

    for field in data.header.model_fields:
        print(field)



    print(data.header)