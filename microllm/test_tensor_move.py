from tinygrad.tensor import Tensor
from tinygrad.dtype import dtypes

def nonzero(tensor: Tensor):
    # Get indices of non-zero elements
    # Use the mask to filter indices
    mask = Tensor.where(tensor != 0, Tensor([1], dtype=dtypes.int32), 0)
    indices = Tensor.arange(tensor.shape[0]) * mask
    print("Indices:", indices.numpy())
    print("Mask:", mask.numpy())

    test = Tensor.gather(tensor, indices, dim=0)
    print("Test:", test.numpy())

    return tensor

# Example usage
tensor = Tensor([0, 0, 3, 0, 6, 0])
result = nonzero(tensor)
print("Result:", result.numpy()) # [3, 6]