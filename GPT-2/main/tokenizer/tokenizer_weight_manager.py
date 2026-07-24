import os
from tokenizers import Tokenizer as HFTokenizer

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class TokenizerWeightsManager:
    def __init__(self, tokenizer, save_tokenizer_weights=False, save_tokenizer_path=f"{base_dir}/weights"):
        self.tokenizer = tokenizer
        self.save_tokenizer_weights = save_tokenizer_weights
        self.save_tokenizer_path = save_tokenizer_path

    def save_weights(self):
        if not self.save_tokenizer_weights:
            return
        os.makedirs(self.save_tokenizer_path, exist_ok=True)
        weight_path = os.path.join(self.save_tokenizer_path, 'tokenizer_weights.json')
        
        # Hugging Face saves the merges, vocab, and config all in one file
        self.tokenizer._hf_tokenizer.save(weight_path)
    
    def load_weights(self):
        weight_path = os.path.join(self.save_tokenizer_path, 'tokenizer_weights.json')
        if os.path.exists(weight_path):
            # Load weights and overwrite the wrapper's tokenizer instance
            self.tokenizer._hf_tokenizer = HFTokenizer.from_file(weight_path)
