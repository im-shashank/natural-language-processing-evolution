import torch.nn as nn

class FlattenConsecutive(nn.Module):
    """
    A module that groups consecutive elements in a sequence and flattens them.
    
    This layer is used to compress the sequence dimension by grouping consecutive
    elements together. It's particularly useful in WaveNet architectures where
    we want to reduce sequence length while maintaining feature information.
    
    Attributes:
        n (int): Number of consecutive elements to group together
    """

    def __init__(self, n, device='cpu'):
        """
        Initializes the FlattenConsecutive layer.
        
        Args:
            n (int): Number of consecutive elements to group together
            device (str): Device to place tensors on ('cpu', 'cuda', 'mps')
        """
        super().__init__()
        self.n = n
        
    def forward(self, x):
        """
        Performs the forward pass through the layer.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, T, C) where B=batch, T=sequence, C=channels
            
        Returns:
            torch.Tensor: Output tensor with flattened consecutive elements
        """
        B, T, C = x.shape
        # Group 'n' consecutive elements in the sequence dimension
        x = x.view(B, T // self.n, C * self.n)
        
        # If the sequence length becomes 1, squeeze it out so it works with nn.Linear
        if x.shape[1] == 1:
            x = x.squeeze(1)
            
        return x