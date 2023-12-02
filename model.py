import numpy as np
from tinygrad.tensor import Tensor
from tinygrad.helpers import dtypes
from tinygrad.nn import Linear, Embedding
from typing import List
from file_type import gguf

data, file = gguf.read_gguf("E:\LLM\models\TheBloke\Mistral-7B-Instruct-v0.1-GGUF\mistral-7b-instruct-v0.1.Q4_0.gguf")
print(file)