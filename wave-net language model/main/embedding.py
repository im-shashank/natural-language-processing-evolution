import torch
import torch.nn as nn

class Embedding(nn.Module):
    """
    A simple embedding layer for converting character indices to dense vectors.
    
    This embedding layer maps each character index to a dense vector representation
    that can be learned during training. It's used as the first layer in the WaveNet model.
    
    Attributes:
        weights (nn.Parameter): Learnable weight matrix of shape (vocabulary_size, feature_dimension)
    """

    def __init__(self, vocabulary_size, generator, feature_dimension, device='cpu'):
        """
        Initializes the embedding layer with random weights.
        
        Args:
            vocabulary_size (int): Number of unique characters in the vocabulary
            generator (torch.Generator): Random number generator for weight initialization
            feature_dimension (int): Dimension of the embedding vectors
            device (str): Device to place tensors on ('cpu', 'cuda', 'mps')
        """
        super().__init__()
        self.weights = nn.Parameter(
            (
                torch.randn((vocabulary_size, feature_dimension),
                             device=device,
                             generator=generator) * 0.01
            ).requires_grad_(True))

    def forward(self, training_batch):
        """
        Performs the forward pass through the embedding layer.
        
        Args:
            training_batch (torch.Tensor): Input tensor of character indices
            
        Returns:
            torch.Tensor: Embedded representation of input characters
        """
        embedding = self.weights[training_batch]
        return embedding
    