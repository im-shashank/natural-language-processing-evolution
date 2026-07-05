import torch
import torch.nn.functional as F
from tokenizer import *


class EmbeddingLayer:
    """
    Learnable embedding layer that maps discrete character indices to dense vectors.

    Instead of one-hot encoding each character (which would be sparse and wasteful),
    this layer uses a lookup table (embedding matrix) of shape
    (vocabulary_size, feature_dimension). Each unique character gets its own
    trainable vector representation.

    During training, the embedding vectors are updated via backpropagation so that
    characters appearing in similar contexts develop similar representations.

    Attributes:
        device (str): The device to create tensors on ('cpu', 'cuda', 'mps').
        generator (torch.Generator): Seeded random number generator for reproducibility.
        words (list): List of words used to build the vocabulary.
        tokenizer (Tokenizer): Tokenizer instance for character-to-index conversion.
        feature_dimension (int): Dimensionality of each character's embedding vector.
        vocabulary_size (int): Total number of unique characters in the vocabulary.
        C (torch.Tensor): The embedding lookup table of shape (vocabulary_size, feature_dimension).
    """

    def __init__(self, device='cpu', tokenizer=None, feature_dimension=2, words=None, generator=None):
        """
        Initializes the embedding layer with a randomly-initialized lookup table.

        Args:
            device (str): The device to create tensors on. Defaults to 'cpu'.
            tokenizer (Tokenizer, optional): Tokenizer instance for character-to-index conversion.
                If None, a new Tokenizer is created from the words list.
            feature_dimension (int): Dimensionality of each character's embedding vector.
                Defaults to 2.
            words (list, optional): List of words used to build the vocabulary.
                If None, reads from 'bi-gram language model/resources/names.txt'.
            generator (torch.Generator, optional): Seeded random number generator for reproducibility.
                If None, creates a new generator with seed 2147483647.
        """
        self.device = device
        self.generator = torch.Generator(device=self.device).manual_seed(2147483647) if generator == None else generator
        self.words = open('bi-gram language model/resources/names.txt', 'r').read().splitlines() if words == None else words
        self.tokenizer = Tokenizer(self.words) if tokenizer == None else tokenizer
        self.feature_dimension = feature_dimension
        self.vocabulary_size = len(self.tokenizer.stoi)
        self.C = (torch.randn((self.vocabulary_size, self.feature_dimension), 
                             device=self.device,
                             generator=self.generator) * 0.01).requires_grad_(True)
    
    def __call__(self, X):
        """
        Performs a forward pass through the embedding layer.

        Looks up the embedding vector for each character index in the input tensor.
        This is equivalent to multiplying a one-hot encoded representation by the
        embedding matrix C, but implemented as an efficient indexing operation.

        Args:
            X (torch.Tensor): Input tensor of character indices with shape
                (batch_size, context_length) or (context_length,).

        Returns:
            torch.Tensor: Embedding tensor of shape (batch_size, context_length, feature_dimension)
                or (context_length, feature_dimension), matching the input shape with an
                additional feature dimension appended.
        """
        embedding = self.C[X]
        return embedding