import torch
import torch.nn.functional as F
from tokenizer import *


class EmbeddingLayer:

    def __init__(self, device='cpu', tokenizer=None, context_lenght=3, feature_dimension=2, words=None):
        self.words = open('bi-gram language model/resources/names.txt', 'r').read().splitlines() if words == None else words
        self.tokenizer = Tokenizer(self.words) if tokenizer == None else tokenizer
        self.device = device
        self.context_length = context_lenght
        self.feature_dimension = feature_dimension
        self.vocabulary_size = len(self.tokenizer.stoi)
        self.C = torch.randn((self.vocabulary_size, self.feature_dimension), requires_grad=True, device=self.device)

    def __call__(self):
        block_size = self.context_length # context length: how many characters do we take to predict the next one?
        X, Y = [], []
        for w in self.words:
            context = [0] * block_size
            for ch in w + '.':
                ix = self.tokenizer.stoi[ch]
                X.append(context)
                Y.append(ix)
                # print(''.join(self.tokenizer.itos[i] for i in context), '--->', self.tokenizer.itos[ix])
                context = context[1:] + [ix] # crop and append
        
        X = torch.tensor(X, device=self.device)
        Y = torch.tensor(Y, device=self.device)

        embedding = self.C[X]

        return [X, Y, embedding]