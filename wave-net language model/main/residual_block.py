import torch.nn as nn
from batch_norm_1d import *

class ResidualBlock(nn.Module):
    """
    A deep learning residual block that maintains tensor dimensions.
    It creates an 'express lane' for gradients to flow backward.
    """
    def __init__(self, hidden_dimension, dropout_prob, device='cpu'):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dimension, hidden_dimension, device=device, bias=False),
            BatchNorm1d(hidden_dimension, device=device),
            nn.GELU(),
            nn.Dropout(p=dropout_prob)
        )

    def forward(self, x):
        # The Residual Formula: Output = Transformation(x) + x
        return self.net(x) + x