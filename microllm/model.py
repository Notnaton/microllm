from tinygrad.tensor import Tensor
import numpy as np

def mod(tensor: Tensor, M):
    div_result = tensor / M  # Regular division
    floored_result = div_result.floor()  # Flooring the division result to mimic floor division
    multiplied_back = floored_result * M
    return tensor - multiplied_back

def filter(tensor: Tensor, filter: int):
    mask = tensor != filter
    return tensor[mask]

def find_pairs_tinygrad(N, T):
    # Create tensors for a and b ranging from 0 to N-1
    a = Tensor.arange(N).reshape((N, 1))  # Column vector
    b = Tensor.arange(N).reshape((1, N))  # Row vector

    # Compute the outer product and then take modulus N
    product = a * b
    prod_mod = mod(product, N)

    # Find where product mod N equals T
    mask = prod_mod == T

    # Apply mask to a and b
    mask_a = Tensor.where(mask, a, 0)
    mask_b = Tensor.where(mask, b, 0)

    # Now mask_a and mask_b have the values of a and b where (a * b) mod N == T, and 0 everywhere else
    # You can use these tensors in subsequent computations, and the zeros will not affect the result

    return mask_a, mask_b

    #sum = mask.sum()
    #ret = Tensor.empty(sum)

    #test = Tensor.where(mask == 1, a, 0)
    #print(test.flatten().numpy())
    #return test.numpy()

# Example usage
N = 2**8 # Replace with your N
T = 9   # Replace with your T
result = find_pairs_tinygrad(N, T)
#print("Pairs (a, b) such that a*b % N == T:", result)





"""
# Define size and target modulo
N = 256
T = 5

# Create tensors for a and b
a = Tensor.arange(N).reshape((N, 1))  # Column vector of [0, 1, ..., N-1]
b = Tensor.arange(N).reshape((1, N))  # Row vector of [0, 1, ..., N-1]

# Compute the product matrix
product_matrix = a * b

# Store a and b in two separate tensors (to simulate stacking)
a_matrix = a * Tensor.ones((1, N))  # Replicate 'a' across columns
b_matrix = b * Tensor.ones((N, 1))  # Replicate 'b' across rows

# Calculate modulo
mod_matrix = product_matrix % N

# Create a mask where product modulo N equals T
mask = Tensor.where(mod_matrix == T, Tensor.ones(mod_matrix.shape), Tensor.zeros(mod_matrix.shape))

# Apply mask to the stored 'a' and 'b' matrices
filtered_a = a_matrix * mask
filtered_b = b_matrix * mask

# These matrices now hold the values of 'a' and 'b' wherever the condition (a * b % N == T) was met, and zeros elsewhere

#523 x 863 = 451349 M
mod_matrix = product_matrix - multiplied_back

# Print the resulting matrix after the manual modulo operation
print(mod_matrix.numpy())

matches = Tensor.where(mod_matrix == T, 1, 0)
matches_np = matches.numpy()  # Convert to numpy array for further processing
indices = np.argwhere(matches_np == 1)

print(matches.numpy())

#523 x 863 = 451349
# n1 c2 = a1*b1 % 256 | ((a1*b1)-((a1*b1)%256)) / 256"""
"""
c1 = 0
n1 = 9
target n = n1 - c1 wrapping = 9
a1 = [1,3,9]
b1 = [9,3,1]

c2 = (a1*b1 +c1) // 256 div plus .floor()
n2 = 4
target n = 4 - 0

"""