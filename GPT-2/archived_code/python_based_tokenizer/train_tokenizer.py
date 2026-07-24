import os
import sys

from datasets import load_dataset # huggingface datasets

# Add the 'main' directory to sys.path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tokenizer import Tokenizer

class TrainTokenizer:
    def __init__(self, num_samples=10000):
        # We pass save_tokenizer_weights=True to actually save the weights after training
        self.tokenizer = Tokenizer(
            desired_vocab_size=50257,
            save_tokenizer_weights=True,
            training_cutoff_metric="desired_vocab_size"
            )

        # We use streaming=True so we don't have to download the 54GB dataset all at once.
        # openwebtext contains dictionaries where the actual string is under the "text" key.
        self.dataset = load_dataset("Skylion007/openwebtext", split="train", streaming=True)
        self.num_samples = num_samples
    
    def __call__(self):
        print(f"Accumulating {self.num_samples} samples from openwebtext...")
        text_data = []
        
        # Iterate over the dataset and extract the text
        for i, sample in enumerate(self.dataset):
            if i >= self.num_samples:
                break
            text_data.append(sample["text"])
            
        # Join the accumulated text into a single large string
        full_text = "\n".join(text_data)
        
        print(f"Training tokenizer on {len(full_text)} characters...")
        self.tokenizer.train(full_text)
        print("Training complete!")

if __name__ == "__main__":
    trainer = TrainTokenizer()
    trainer()