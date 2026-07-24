import os
import sys
from datasets import load_dataset

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tokenizer import Tokenizer

class TrainTokenizer:
    def __init__(self, num_samples=1000000):
        self.tokenizer = Tokenizer(
            desired_vocab_size=50304,
            save_tokenizer_weights=True
        )
        self.dataset = load_dataset("Skylion007/openwebtext", split="train", streaming=True)
        self.num_samples = num_samples
    
    def dataset_iterator(self):
        """A generator that yields text chunks one at a time."""
        for i, sample in enumerate(self.dataset):
            if i >= self.num_samples:
                break
            yield sample["text"]
            
    def __call__(self):
        print(f"Training tokenizer on {self.num_samples} samples from openwebtext...")
        # Train directly from the generator to save memory
        self.tokenizer.train_from_iterator(self.dataset_iterator())
        print("Training complete!")

if __name__ == "__main__":
    trainer = TrainTokenizer()
    trainer()
