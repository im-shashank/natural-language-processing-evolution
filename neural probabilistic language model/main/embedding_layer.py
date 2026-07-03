import torch
import torch.nn.functional as F
from tokenizer import *


class EmbeddingLayer:

    def __init__(self, device='cpu', tokenizer=None, feature_dimension=2, words=None, generator=None):
        self.device = device
        self.generator = torch.Generator(device=self.device).manual_seed(2147483647) if generator == None else generator
        self.words = open('bi-gram language model/resources/names.txt', 'r').read().splitlines() if words == None else words
        self.tokenizer = Tokenizer(self.words) if tokenizer == None else tokenizer
        self.feature_dimension = feature_dimension
        self.vocabulary_size = len(self.tokenizer.stoi)
        self.C = torch.randn((self.vocabulary_size, self.feature_dimension), 
                             requires_grad=True, 
                             device=self.device,
                             generator=self.generator)
    
    def __call__(self, X):
        embedding = self.C[X]
        return embedding