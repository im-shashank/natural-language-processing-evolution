import torch
import torch.nn.functional as F
from tokenizer import *

class CreateDataset:

    def __init__(self, device='cpu', tokenizer=None, context_lenght=3, words=None, generator=None):
        self.device = device
        self.generator = torch.Generator(device=self.device).manual_seed(2147483647) if generator == None else generator
        self.words = open('bi-gram language model/resources/names.txt', 'r').read().splitlines() if words == None else words
        self.tokenizer = Tokenizer(self.words) if tokenizer == None else tokenizer
        self.context_length = context_lenght

    def __call__(self, words=None):
        if words == None:
            print("the words dataset is null")
            pass
        block_size = self.context_length # context length: how many characters do we take to predict the next one?
        X, Y = [], []
        try:
            for w in words:
                context = [0] * block_size
                for ch in w + '.':
                    ix = self.tokenizer.stoi[ch]
                    X.append(context)
                    Y.append(ix)
                    # print(''.join(self.tokenizer.itos[i] for i in context), '--->', self.tokenizer.itos[ix])
                    context = context[1:] + [ix] # crop and append
        
            X = torch.tensor(X, device=self.device)
            Y = torch.tensor(Y, device=self.device)
        except Exception as e:
            print(f"Failed to create dataset with exception: {e}")

        return [X, Y]