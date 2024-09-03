from tinygrad.tensor import Tensor
from tinygrad.nn import Linear, LayerNorm
import numpy as np

class PhiAttention:
    def __init__(self, config):
        self.num_heads = config['num_attention_heads']
        self.hidden_size = config['hidden_size']
        self.head_size = self.hidden_size // self.num_heads
        self.qkv = Linear(self.hidden_size, 3 * self.hidden_size, bias=True)
        self.out_proj = Linear(self.hidden_size, self.hidden_size, bias=True)

    def __call__(self, x):
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q, k, v = [t.reshape(t.shape[0], t.shape[1], self.num_heads, self.head_size).transpose(1, 2) for t in (q, k, v)]
        
        attn = (q @ k.transpose(-2, -1) / np.sqrt(self.head_size)).softmax(axis=-1)
        return self.out_proj((attn @ v).transpose(1, 2).reshape(x.shape[0], x.shape[1], self.hidden_size))

class PhiMLP:
    def __init__(self, config):
        self.fc1 = Linear(config['hidden_size'], config['intermediate_size'], bias=True)
        self.fc2 = Linear(config['intermediate_size'], config['hidden_size'], bias=True)

    def __call__(self, x):
        return self.fc2(self.fc1(x).gelu())

class PhiLayer:
    def __init__(self, config):
        self.attention = PhiAttention(config)
        self.mlp = PhiMLP(config)
        self.ln1 = LayerNorm(config['hidden_size'])
        self.ln2 = LayerNorm(config['hidden_size'])

    def __call__(self, x):
        x = x + self.attention(self.ln1(x))
        return x + self.mlp(self.ln2(x))

class PhiModel:
    def __init__(self, config):
        self.embed = Tensor.uniform(config['vocab_size'], config['hidden_size'])
        self.layers = [PhiLayer(config) for _ in range(config['num_hidden_layers'])]
        self.output_norm = LayerNorm(config['hidden_size'])
        self.output = Linear(config['hidden_size'], config['vocab_size'], bias=True)

    def __call__(self, input_ids):
        x = self.embed[input_ids]
        for layer in self.layers:
            x = layer(x)
        return self.output(self.output_norm(x))

def load_phi_model(gguf_data):
    metadata = gguf_data['metadata']
    tensor_data = gguf_data['tensor_data']
    
    config = {
        'vocab_size': int(metadata['phi3.embedding_length']),
        'hidden_size': int(metadata['phi3.embedding_length']),
        'num_hidden_layers': int(metadata['phi3.block_count']),
        'num_attention_heads': int(metadata['phi3.attention.head_count']),
        'intermediate_size': int(metadata['phi3.feed_forward_length']),
    }
    
    model = PhiModel(config)
    
    def load_tensor(name):
        tensor = tensor_data.get(name)
        return Tensor(tensor) if tensor is not None else None

    model.embed = load_tensor('model.embed_tokens.weight') or model.embed

    for i, layer in enumerate(model.layers):
        prefix = f'model.layers.{i}.'
        
        # Attention weights
        layer.attention.qkv.weight = load_tensor(f'{prefix}self_attn.q_proj.weight') or layer.attention.qkv.weight
        layer.attention.qkv.bias = load_tensor(f'{prefix}self_attn.q_proj.bias') or layer.attention.qkv.bias
        
        layer.attention.out_proj.weight = load_tensor(f'{prefix}self_attn.out_proj.weight') or layer.attention.out_proj.weight
        layer.attention.out_proj.bias = load_tensor(f'{prefix}self_attn.out_proj.bias') or layer.attention.out_proj.bias
        
        # MLP weights
        layer.mlp.fc1.weight = load_tensor(f'{prefix}mlp.fc1.weight') or layer.mlp.fc1.weight
        layer.mlp.fc1.bias = load_tensor(f'{prefix}mlp.fc1.bias') or layer.mlp.fc1.bias
        
        layer.mlp.fc2.weight = load_tensor(f'{prefix}mlp.fc2.weight') or layer.mlp.fc2.weight
        layer.mlp.fc2.bias = load_tensor(f'{prefix}mlp.fc2.bias') or layer.mlp.fc2.bias
        
        # Layer norm weights
        layer.ln1.weight = load_tensor(f'{prefix}input_layernorm.weight') or layer.ln1.weight
        layer.ln1.bias = load_tensor(f'{prefix}input_layernorm.bias') or layer.ln1.bias
        
        layer.ln2.weight = load_tensor(f'{prefix}post_attention_layernorm.weight') or layer.ln2.weight
        layer.ln2.bias = load_tensor(f'{prefix}post_attention_layernorm.bias') or layer.ln2.bias
    
    return model

def run_inference(model, input_ids):
    if isinstance(input_ids, Tensor):
        return model(input_ids).softmax(axis=-1)
    else:
        return model(Tensor(input_ids)).softmax(axis=-1)