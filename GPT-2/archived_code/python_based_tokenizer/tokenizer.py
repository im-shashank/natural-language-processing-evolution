import regex as re
from tqdm import tqdm

from python_based_tokenizer.tokenizer_weight_manager import TokenizerWeightsManager

class Tokenizer:
    """A byte-level Byte Pair Encoding (BPE) tokenizer.
    
    This tokenizer separates training (building the vocabulary) from 
    encoding/decoding, ensuring that token IDs have a static meaning.
    """

    def __init__(self, compression_ratio=1.5, desired_vocab_size=50257, training_cutoff_metric="compression_ratio", save_tokenizer_weights=False):
        """Initializes the Tokenizer with a target compression ratio.
        
        Args:
            compression_ratio (float): The target ratio of original byte length 
                to compressed token length before stopping training.
            desired_vocab_size (int): The desired size of the vocabulary.
            training_cutoff_metric (str): The metric used to determine when to stop training.
                Options: "compression_ratio", "desired_vocab_size"
        """
        self.regex_pattern = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}++| ?\p{N}++| ?[^\s\p{L}\p{N}]++|\s++$|\s+(?!\S)|\s""")
        self.compression_ratio = compression_ratio
        self.desired_vocab_size = desired_vocab_size
        self.training_cutoff_metric = training_cutoff_metric

        # Maps (token1, token2) -> new_token_id (used for encoding)
        self.merges = {}

        # Maps token_id -> bytes (used for decoding)
        self.vocab = {idx: bytes([idx]) for idx in range(256)}

        self.weight_manager = TokenizerWeightsManager(self, save_tokenizer_weights=save_tokenizer_weights)
        self.weight_manager.load_weights()

    def __call__(self, text):
        """Encodes the input text and returns the compressed BPE token list.
        
        Args:
            text (str): The input text to tokenize.
            
        Returns:
            list[int]: The list of encoded BPE token IDs.
        """
        return self.encode(text)

    def regex_findall(self, text):
        """Uses regex to find all the words, spaces and punctuations."""
        return re.findall(self.regex_pattern, text)

    def train(self, text):
        """Trains the tokenizer on the given text to build the vocabulary.
        
        Iteratively identifies and merges the most common adjacent token pairs 
        until the target compression ratio is met, storing these rules statically.
        
        Args:
            text (str): The training corpus.
        """
        text_chunks = self.regex_findall(text)
        chunk_tokens = [list(map(int, chunk.encode("utf-8"))) for chunk in text_chunks]
        
        original_tokens_length = sum(len(tokens) for tokens in chunk_tokens)
        new_token_id = 256
        
        pbar = tqdm(total=self.desired_vocab_size - 256 if self.training_cutoff_metric == "desired_vocab_size" else None, desc="Training Tokenizer")
        
        while True:
            top_pair = self._get_top_pair(chunk_tokens)
            if not top_pair:
                break
                
            self.merges[top_pair] = new_token_id
            self.vocab[new_token_id] = self.vocab[top_pair[0]] + self.vocab[top_pair[1]]
            
            chunk_tokens = [self._merge_pair(tokens, top_pair, new_token_id) for tokens in chunk_tokens]
            new_token_id += 1
            
            if pbar:
                pbar.update(1)
            else:
                if len(self.vocab) % 100 == 0:
                    print(f"Training Progress | Vocab Size: {len(self.vocab)}")
            
            if self.training_cutoff_metric == "compression_ratio":
                new_token_length = sum(len(tokens) for tokens in chunk_tokens)
                compression_ratio = original_tokens_length / new_token_length
                if pbar:
                    pbar.set_postfix({"compression_ratio": f"{compression_ratio:.3f}"})
                if compression_ratio >= self.compression_ratio:
                    break
            elif self.training_cutoff_metric == "desired_vocab_size":
                if len(self.vocab) >= self.desired_vocab_size:
                    break
                    
        if pbar:
            pbar.close()

        self.weight_manager.save_weights()

    def encode(self, text):
        """Encodes text into a list of BPE token IDs using the trained vocabulary.
        
        Args:
            text (str): The input text to encode.
            
        Returns:
            list[int]: A list of BPE token IDs.
        """
        text_chunks = self.regex_findall(text)
        final_tokens = []
        
        for chunk in text_chunks:
            tokens = list(map(int, chunk.encode("utf-8")))
            
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
                
            final_tokens.extend(tokens)
            
        return final_tokens

    def decode(self, encoded):
        """Decodes BPE token IDs back into a human-readable UTF-8 string.
        
        Args:
            encoded (list[int]): The list of BPE token IDs to decode.
            
        Returns:
            str: The decoded text, with invalid UTF-8 bytes replaced.
        """
        b = b"".join(self.vocab[token_id] for token_id in encoded)
        return b.decode("utf-8", errors='replace')

    def _get_top_pair(self, chunk_tokens):
        """Finds and returns the most frequent adjacent pair of tokens across all chunks.
        
        Args:
            chunk_tokens (list[list[int]]): The current list of token ID lists for each chunk.
            
        Returns:
            tuple[int, int] | None: The most frequent adjacent token pair, or None if no pairs exist.
        """
        pairs = {}
        for tokens in chunk_tokens:
            for pair in zip(tokens, tokens[1:]):
                pairs[pair] = pairs.get(pair, 0) + 1
        return max(pairs, key=pairs.get) if pairs else None
        
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