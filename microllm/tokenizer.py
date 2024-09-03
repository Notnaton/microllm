from typing import List

class Tokenizer:
    def __init__(self, gguf_data):
        self.vocab = self._load_vocab(gguf_data)
        self.ids_to_tokens = {v: k for k, v in self.vocab.items()}
        self.unk_token_id = gguf_data['metadata'].get('tokenizer.ggml.unknown_token_id', 0)
        self.max_token_length = max(len(token) for token in self.vocab.keys())
        self.space_token = ' '
        self.space_token_id = self.vocab.get(self.space_token, self.unk_token_id)

    def _load_vocab(self, gguf_data):
        tokens = gguf_data['metadata'].get('tokenizer.ggml.tokens')
        if not tokens:
            raise ValueError("Tokenizer vocabulary not found in GGUF file")
        
        return {token: i for i, token in enumerate(tokens)}

    def encode(self, text: str) -> List[int]:
        tokens = []
        i = 0
        while i < len(text):
            if text[i] == ' ':
                tokens.append(self.space_token_id)
                i += 1
                continue

            longest_match = None
            for j in range(min(self.max_token_length, len(text) - i), 0, -1):
                substr = text[i:i+j]
                if substr in self.vocab:
                    longest_match = substr
                    break
            if longest_match:
                tokens.append(self.vocab[longest_match])
                i += len(longest_match)
            else:
                tokens.append(self.unk_token_id)
                i += 1
        return tokens

    def decode(self, ids: List[int]) -> str:
        return ''.join(self.ids_to_tokens.get(id, chr(id)) if id != self.space_token_id else ' ' for id in ids)

def load_tokenizer(gguf_data) -> Tokenizer:
    return Tokenizer(gguf_data)

# Example usage
if __name__ == '__main__':
    from microllm.read_gguf import gguf_file
    
    gguf_path = '/home/anton/.cache/lm-studio/models/lmstudio-community/Phi-3.5-mini-instruct-GGUF/Phi-3.5-mini-instruct-Q8_0.gguf'
    with open(gguf_path, 'rb') as file:
        gguf = gguf_file(file)
    
    gguf_dict = gguf.to_dict()
    
    print("Available metadata keys:")
    for key in gguf_dict['metadata']:
        print(f"- {key}")
    
    tokenizer = load_tokenizer(gguf_dict)
    
    # Test encoding and decoding
    test_text = "Hello, how are you?"
    encoded = tokenizer.encode(test_text)
    decoded = tokenizer.decode(encoded)
    
    print(f"\nOriginal: {test_text}")
    print(f"Encoded: {encoded}")
    print(f"Decoded: {decoded}")

    # Print vocabulary size
    print(f"\nVocabulary size: {len(tokenizer.vocab)}")

    # Print some example tokens
    print("\nExample tokens:")
    for i, (token, id) in enumerate(list(tokenizer.vocab.items())[:10]):
        print(f"{id}: {repr(token)}")