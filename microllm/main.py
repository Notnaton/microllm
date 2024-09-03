from microllm.read_gguf import gguf_file
from microllm.phi_model import load_phi_model, run_inference
from microllm.tokenizer import load_tokenizer
from tinygrad.tensor import Tensor
import numpy as np

def top_k_sampling(logits, k=10, temperature=0.7):
    logits = logits.numpy()  # Convert to numpy array
    logits = logits / temperature
    top_k_indices = np.argpartition(logits, -k)[-k:]
    top_k_logits = logits[top_k_indices]
    probs = np.exp(top_k_logits) / np.sum(np.exp(top_k_logits))
    choice = np.random.choice(top_k_indices, p=probs)
    return choice

# Load GGUF file
gguf_path = '/home/anton/.cache/lm-studio/models/lmstudio-community/Phi-3.5-mini-instruct-GGUF/Phi-3.5-mini-instruct-Q8_0.gguf'
with open(gguf_path, 'rb') as file:
    gguf = gguf_file(file)

gguf_dict = gguf.to_dict()

print("GGUF file loaded successfully")
print("Number of tensors:", len(gguf.tensor_data))

# Load Phi model
try:
    model = load_phi_model(gguf_dict)
    print("\nPhi model loaded successfully")
except Exception as e:
    print(f"Error loading Phi model: {str(e)}")
    raise

# Load tokenizer
try:
    tokenizer = load_tokenizer(gguf_dict)
    print("Tokenizer loaded successfully")
except Exception as e:
    print(f"Error loading tokenizer: {str(e)}")
    raise

# Prepare input
input_text = "Hello, how are you?"
input_ids = tokenizer.encode(input_text)
input_tensor = Tensor([input_ids])

print("\nInput text:", input_text)
print("Input IDs:", input_ids)
print("Input tensor shape:", input_tensor.shape)

# Run inference
try:
    output = run_inference(model, input_tensor)
    next_token_id = top_k_sampling(output[0, -1])
    next_token = tokenizer.decode([next_token_id])
    print(f"\nNext token: {next_token}")
    
    # Generate a longer sequence
    generated_text = input_text
    max_length = 100
    eos_token_id = tokenizer.encode('.')[0]  # Assuming '.' is the end of sentence token
    
    for _ in range(max_length):
        output = run_inference(model, Tensor([tokenizer.encode(generated_text)]))
        next_token_id = top_k_sampling(output[0, -1])
        if next_token_id == eos_token_id:
            break
        next_token = tokenizer.decode([next_token_id])
        generated_text += next_token
        if len(generated_text) >= max_length:
            break
    
    print(f"\nGenerated text: {generated_text}")
except Exception as e:
    print(f"Error during inference: {str(e)}")
    raise