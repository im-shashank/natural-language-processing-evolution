import torch

from tokenizer import *

torch.manual_seed(1337)

class CreateDataset:

    def __init__(self, tokenizer, device='cpu'):
        self.device = device
        self.tokenizer = tokenizer

    def __call__(self, text):
        self.dataset = torch.tensor(self.tokenizer.encode(text), dtype=torch.long, device=self.device)

        return self.dataset
    