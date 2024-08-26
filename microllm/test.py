from tinygrad import Tensor
from tinygrad.nn import Linear
import math

class SimpleRWKV:
    def __init__(self, d_model, context_length):
        self.d_model = d_model
        self.context_length = context_length
        
        # Initialize layers
        self.time_mix_k = Tensor.uniform(d_model)
        self.time_mix_v = Tensor.uniform(d_model)
        self.time_mix_r = Tensor.uniform(d_model)
        self.time_decay = Tensor.uniform(d_model)
        
        self.key = Linear(d_model, d_model)
        self.value = Linear(d_model, d_model)
        self.receptance = Linear(d_model, d_model)
        self.output = Linear(d_model, d_model)

    def forward(self, x):
        B, T, C = x.shape
        
        # Initialize state
        state_k = Tensor.zeros(B, C)
        state_v = Tensor.zeros(B, C)
        
        outputs = []
        
        for t in range(T):
            # Current input
            xt = x[:, t, :]
            
            # Time-mixing
            k = self.key(xt * self.time_mix_k + state_k * (1 - self.time_mix_k))
            v = self.value(xt * self.time_mix_v + state_v * (1 - self.time_mix_v))
            r = self.receptance(xt * self.time_mix_r + state_k * (1 - self.time_mix_r))
            
            # Update state
            state_k = k
            state_v = v
            
            # Compute output
            y = self.output(v * (r.sigmoid()))
            outputs.append(y)
        
        return Tensor.stack(outputs, dim=1)

# Example usage
d_model = 256
context_length = 1024
batch_size = 32
seq_length = 512

model = SimpleRWKV(d_model, context_length)
input_tensor = Tensor.uniform(batch_size, seq_length, d_model)
output = model.forward(input_tensor)
print(output.shape)  # Should be (batch_size, seq_length, d_model)