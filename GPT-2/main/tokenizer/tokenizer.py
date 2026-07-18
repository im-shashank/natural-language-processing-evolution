import os

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class Tokenizer:

    def __init__(self, compression_ratio=1.5):
        self.compression_ratio = compression_ratio
        self.tokens_map = {}

    def __call__(self, text):
        encoded = self.encode(text)
        return encoded

    def encode(self, text):
        tokens = list(map(int, text.encode("utf-8")))
        original_tokens_length = len(tokens)
        new_token_id = 256
        while True:
            top_pair = self._get_top_pair(tokens)
            tokens = self._merge_pair(tokens, top_pair, new_token_id)
            new_token_id += 1
            new_token_length = len(tokens)
            compression_ratio = original_tokens_length / new_token_length
            if compression_ratio >= self.compression_ratio:
                break
        return tokens

    def decode(self, encoded):
        decompressed_token = list(encoded)

        while any(token in self.tokens_map for token in decompressed_token):
            new_tokens = []
            for token in decompressed_token:
                if token in self.tokens_map:
                    pair = self.tokens_map[token]
                    new_tokens.append(pair[0])
                    new_tokens.append(pair[1])
                else:
                    new_tokens.append(token)
            
            decompressed_token = new_tokens
        
        decoded = bytes(decompressed_token).decode("utf-8")
        return decoded


    def _get_top_pair(self, tokens):
        pairs = {}

        for token in zip(tokens, tokens[1:]):
            pairs[token] = pairs.get(token, 0) + 1
        
        top_pair = max(pairs, key=pairs.get)
        return top_pair
        
    def _merge_pair(self, tokens, top_pair, new_token_id):
        new_tokens = []
        i = 0
        while i < len(tokens):
            if i < len(tokens) - 1 and tokens[i] == top_pair[0] and tokens[i+1] == top_pair[1]:
                new_tokens.append(new_token_id)
                self.tokens_map[new_token_id] = top_pair
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        return new_tokens