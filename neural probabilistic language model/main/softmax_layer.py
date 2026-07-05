"""
Softmax output layer for the Neural Probabilistic Language Model.

This module contains the final layer of the network that maps hidden representations
to character-level logit scores. These logits are then passed through a softmax
function (during training via cross-entropy loss, or explicitly during sampling)
to produce a probability distribution over the vocabulary.
"""

import torch
from tokenizer import *

class SoftmaxLayer:
    """
    Output layer that maps hidden representations to character-level logit scores.

    This is the final layer of the network. It takes the hidden layer output and
    produces raw logit scores for every character in the vocabulary. Higher logits
    indicate higher predicted probability for each character.

    The softmax function (applied externally via F.softmax or F.cross_entropy)
    converts these logits into a proper probability distribution that sums to 1.

    Attributes:
        device (str): The device to create tensors on ('cpu', 'cuda', 'mps').
        generator (torch.Generator): Seeded random number generator for reproducibility.
        tokenizer (Tokenizer): Tokenizer instance for vocabulary size determination.
        vocabulary_size (int): Total number of unique characters in the vocabulary.
        num_neron (int): Number of neurons from the hidden layer (input dimension).
        W (torch.Tensor): Weight matrix of shape (num_neuron, vocabulary_size).
        B (torch.Tensor): Bias vector of shape (vocabulary_size,).
    """

    def __init__(self, device='cpu', num_neuron=100, tokenizer=None, generator=None):
        """
        Initializes the softmax layer with randomly-initialized weights and biases.

        Args:
            device (str): The device to create tensors on. Defaults to 'cpu'.
            num_neuron (int): Number of neurons from the hidden layer (input dimension).
                Defaults to 100.
            tokenizer (Tokenizer, optional): Tokenizer instance for vocabulary size determination.
                If None, a new Tokenizer is created from 'bi-gram language model/resources/names.txt'.
            generator (torch.Generator, optional): Seeded random number generator for reproducibility.
                If None, creates a new generator with seed 2147483647.
        """
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
        """
        Performs a forward pass through the softmax layer.

        Computes raw logit scores via a linear transformation:  logits = hidden @ W + B
        These logits represent unnormalized log-probabilities for each character
        in the vocabulary.

        Args:
            hidden_layer (torch.Tensor): Hidden representation tensor of shape
                (batch_size, num_neuron).

        Returns:
            torch.Tensor: Logit scores of shape (batch_size, vocabulary_size),
                where each row contains raw scores for all characters in the vocabulary.
        """
        logits = hidden_layer @ self.W + self.B # calculate logits using matrix multiplication

        return logits