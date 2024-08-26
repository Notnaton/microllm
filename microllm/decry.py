from tinygrad import Tensor

def _mod(tensor: Tensor, M):
    div_result = tensor / M  # Regular division
    floored_result = div_result.floor()  # Flooring the division result to mimic floor division
    multiplied_back = floored_result * M
    return tensor - multiplied_back

def generate_solutions(N, Target):
    # Create tensors for a and b ranging from 0 to N-1
    a = Tensor.arange(N).reshape((N, 1))  # Column vector
    b = Tensor.arange(N).reshape((1, N))  # Row vector

    # Compute the outer product and then take modulus N
    product = a * b
    prod_mod = _mod(product, N)

    # Find where product mod N equals T
    mask_a = Tensor.where(prod_mod == Target, a, 0)
    mask_b = Tensor.where(prod_mod == Target, b, 0)
    solutions = mask_a * mask_b

    return mask_a, mask_b, solutions.squeeze()

def wrapping_add(a, b, N):
    return (a + b) % N

def wrapping_sub(a, b, N):
    return (a - b) % N




N = 10

# 523 x 863 = 451349
input_num = 523 * 863

# split input_num into digits, using N
num, input_num = input_num % N, (input_num - (input_num % N)) // N
digits = [num]
while input_num > 0:
    num, input_num = input_num % N, input_num // N
    digits.append(num)


current_indexes = []
current_data = []

#Generate initial data to run over
for i, n in enumerate(digits):
    current_indexes[i] = 0
    #get c
    

    current_data[i] = generate_solutions(N, n)






