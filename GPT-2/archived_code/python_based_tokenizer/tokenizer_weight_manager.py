import os
import json

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
        
        # JSON requires string keys and doesn't support bytes
        formatted_merges = {f"{k[0]},{k[1]}": v for k, v in self.tokenizer.merges.items()}
        formatted_vocab = {str(k): v.hex() for k, v in self.tokenizer.vocab.items()}
        
        with open(weight_path, 'w', encoding='utf-8') as f:
            json.dump({'merges': formatted_merges, 'vocab': formatted_vocab}, f, indent=4)
    
    def load_weights(self):
        weight_path = os.path.join(self.save_tokenizer_path, 'tokenizer_weights.json')
        if os.path.exists(weight_path):
            with open(weight_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                self.tokenizer.merges = {}
                for k, v in data.get('merges', {}).items():
                    p1, p2 = k.split(',')
                    self.tokenizer.merges[(int(p1), int(p2))] = int(v)
                
                self.tokenizer.vocab = {}
                for k, v in data.get('vocab', {}).items():
                    self.tokenizer.vocab[int(k)] = bytes.fromhex(v)