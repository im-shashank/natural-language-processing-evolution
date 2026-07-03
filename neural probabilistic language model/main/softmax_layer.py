import torch
from tokenizer import *

class SoftmaxLayer:

    def __init__(self, device='cpu', num_neuron=100, tokenizer=None, generator=None):
        self.device = device
        self.generator = torch.Generator(device=self.device).manual_seed(2147483647) if generator == None else generator
        self.tokenizer = Tokenizer(open('bi-gram language model/resources/names.txt', 'r').read().splitlines()) if tokenizer == None else tokenizer
        self.vocabulary_size = len(self.tokenizer.stoi)
        self.num_neron = num_neuron
        self.W = torch.randn((self.num_neron, self.vocabulary_size), 
                             requires_grad=True, 
                             device=self.device, 
                             generator=self.generator)
        self.B = torch.randn(self.vocabulary_size, 
                             requires_grad=True, 
                             device=self.device, 
                             generator=self.generator)
    
    def __call__(self, hidden_layer):
        logits = hidden_layer @ self.W + self.B # calculate logits using matrix multiplication

        return logits