import os

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class Tokenizer:
    """A byte-level Byte Pair Encoding (BPE) tokenizer.
    
    This tokenizer represents text as UTF-8 bytes and iteratively merges the most
    frequent pairs of adjacent tokens to compress the representation.
    """

    def __init__(self, compression_ratio=1.5):
        """Initializes the Tokenizer with a target compression ratio.
        
        Args:
            compression_ratio (float): The target ratio of original byte length 
                to compressed token length before stopping training/encoding.
        """
        self.compression_ratio = compression_ratio
        self.tokens_map = {}

    def __call__(self, text):
        """Encodes the input text and returns the compressed BPE token list.
        
        Args:
            text (str): The input text to tokenize.
            
        Returns:
            list[int]: The list of encoded BPE token IDs.
        """
        encoded = self.encode(text)
        return encoded

    def encode(self, text):
        """Encodes text into a list of BPE token IDs.
        
        Converts text to UTF-8 bytes and iteratively identifies and merges the
        most common adjacent token pairs until the target compression ratio is met.
        
        Args:
            text (str): The input text to encode.
            
        Returns:
            list[int]: A list of BPE token IDs.
        """
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
        """Decodes BPE token IDs back into a human-readable UTF-8 string.
        
        Unpacks merged token IDs recursively by looking up pairs in the reverse
        order they were merged, then converts the final byte sequence back to a string.
        
        Args:
            encoded (list[int]): The list of BPE token IDs to decode.
            
        Returns:
            str: The decoded text, with invalid UTF-8 bytes replaced by .
        """
        decompressed_token = list(encoded)

        for key in self.tokens_map.__reversed__():
            new_tokens = []
            for token in decompressed_token:
                if token == key:
                    new_tokens.extend(self.tokens_map[key])
                else:
                    new_tokens.append(token)
            decompressed_token = new_tokens
        
        decoded = bytes(decompressed_token).decode("utf-8", errors='replace') # replace errorous characters with �
        return decoded


    def _get_top_pair(self, tokens):
        """Finds and returns the most frequent adjacent pair of tokens.
        
        Args:
            tokens (list[int]): The current list of token IDs.
            
        Returns:
            tuple[int, int]: The most frequent adjacent token pair.
        """
        pairs = {}

        for token in zip(tokens, tokens[1:]):
            pairs[token] = pairs.get(token, 0) + 1
        
        top_pair = max(pairs, key=pairs.get)
        return top_pair
        
    def _merge_pair(self, tokens, top_pair, new_token_id):
        """Merges all occurrences of a target token pair into a new token ID.
        
        Args:
            tokens (list[int]): The current list of token IDs.
            top_pair (tuple[int, int]): The token pair to be merged.
            new_token_id (int): The new token ID to replace the pair.
            
        Returns:
            list[int]: The new list of tokens with the pair merged.
        """
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