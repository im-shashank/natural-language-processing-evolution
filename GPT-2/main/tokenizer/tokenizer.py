import os

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class Tokenizer:
    """A byte-level Byte Pair Encoding (BPE) tokenizer.
    
    This tokenizer separates training (building the vocabulary) from 
    encoding/decoding, ensuring that token IDs have a static meaning.
    """

    def __init__(self, compression_ratio=1.5):
        """Initializes the Tokenizer with a target compression ratio.
        
        Args:
            compression_ratio (float): The target ratio of original byte length 
                to compressed token length before stopping training.
        """
        self.compression_ratio = compression_ratio
        # Maps (token1, token2) -> new_token_id (used for encoding)
        self.merges = {}
        # Maps token_id -> bytes (used for decoding)
        self.vocab = {idx: bytes([idx]) for idx in range(256)}

    def __call__(self, text):
        """Encodes the input text and returns the compressed BPE token list.
        
        Args:
            text (str): The input text to tokenize.
            
        Returns:
            list[int]: The list of encoded BPE token IDs.
        """
        return self.encode(text)

    def train(self, text):
        """Trains the tokenizer on the given text to build the vocabulary.
        
        Iteratively identifies and merges the most common adjacent token pairs 
        until the target compression ratio is met, storing these rules statically.
        
        Args:
            text (str): The training corpus.
        """
        tokens = list(map(int, text.encode("utf-8")))
        original_tokens_length = len(tokens)
        new_token_id = 256
        
        while len(tokens) >= 2:
            top_pair = self._get_top_pair(tokens)
            self.merges[top_pair] = new_token_id
            self.vocab[new_token_id] = self.vocab[top_pair[0]] + self.vocab[top_pair[1]]
            
            tokens = self._merge_pair(tokens, top_pair, new_token_id)
            new_token_id += 1
            
            new_token_length = len(tokens)
            compression_ratio = original_tokens_length / new_token_length
            if compression_ratio >= self.compression_ratio:
                break

    def encode(self, text):
        """Encodes text into a list of BPE token IDs using the trained vocabulary.
        
        Args:
            text (str): The input text to encode.
            
        Returns:
            list[int]: A list of BPE token IDs.
        """
        tokens = list(map(int, text.encode("utf-8")))
        
        while len(tokens) >= 2:
            # Find all adjacent pairs in current sequence
            pairs = list(zip(tokens, tokens[1:]))
            
            # Find the pair that was merged earliest (i.e. has the lowest token ID in merges)
            # If none of the pairs are in self.merges, we are done
            eligible_pairs = {pair: self.merges[pair] for pair in pairs if pair in self.merges}
            if not eligible_pairs:
                break
                
            best_pair = min(eligible_pairs, key=eligible_pairs.get)
            new_token_id = eligible_pairs[best_pair]
            
            # Merge the best pair
            tokens = self._merge_pair(tokens, best_pair, new_token_id)
            
        return tokens

    def decode(self, encoded):
        """Decodes BPE token IDs back into a human-readable UTF-8 string.
        
        Args:
            encoded (list[int]): The list of BPE token IDs to decode.
            
        Returns:
            str: The decoded text, with invalid UTF-8 bytes replaced.
        """
        b = b"".join(self.vocab[token_id] for token_id in encoded)
        return b.decode("utf-8", errors='replace')

    def _get_top_pair(self, tokens):
        """Finds and returns the most frequent adjacent pair of tokens.
        
        Args:
            tokens (list[int]): The current list of token IDs.
            
        Returns:
            tuple[int, int]: The most frequent adjacent token pair.
        """
        pairs = {}
        for pair in zip(tokens, tokens[1:]):
            pairs[pair] = pairs.get(pair, 0) + 1
        return max(pairs, key=pairs.get)
        
    def _merge_pair(self, tokens, target_pair, new_token_id):
        """Merges all occurrences of a target token pair into a new token ID.
        
        Args:
            tokens (list[int]): The current list of token IDs.
            target_pair (tuple[int, int]): The token pair to be merged.
            new_token_id (int): The new token ID to replace the pair.
            
        Returns:
            list[int]: The new list of tokens with the pair merged.
        """
        new_tokens = []
        i = 0
        while i < len(tokens):
            if i < len(tokens) - 1 and tokens[i] == target_pair[0] and tokens[i+1] == target_pair[1]:
                new_tokens.append(new_token_id)
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        return new_tokens