import torch
import torch.nn as nn

class BatchNorm1d(nn.Module):
    """
    A 1D batch normalization layer.
    
    This module normalizes the input across the batch dimension, helping to stabilize
    training by reducing internal covariate shift. It maintains running statistics 
    during training for use during inference.
    
    Attributes:
        eps (float): Small constant added to variance for numerical stability
        momentum (float): Momentum factor for updating running statistics
        training (bool): Whether the layer is in training mode
        gamma (nn.Parameter): Scale parameter for normalized values
        beta (nn.Parameter): Shift parameter for normalized values
        running_mean (torch.Tensor): Running mean of batch statistics
        running_var (torch.Tensor): Running variance of batch statistics
    """

    def __init__(self, dim, eps=1e-5, momentum=0.1, device='cpu'):
        """
        Initializes the BatchNorm1d layer.
        
        Args:
            dim (int): Dimension of the input features
            eps (float): Small constant for numerical stability (default: 1e-5)
            momentum (float): Momentum factor for running statistics (default: 0.1)
            device (str): Device to place tensors on ('cpu', 'cuda', 'mps')
        """
        super().__init__()

        self.eps = eps
        self.momentum = momentum
        self.training = True
        # parameters (trained with backprop)
        self.gamma = nn.Parameter(torch.ones(dim, device=device, requires_grad=True))
        self.beta = nn.Parameter(torch.zeros(dim, device=device, requires_grad=True))
        # buffers (trained with a running 'momentum update')
        self.running_mean = torch.zeros(dim, device=device)
        self.running_var = torch.ones(dim, device=device)
  
    def forward(self, x):
        """
        Performs the forward pass through the batch normalization layer.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Normalized output tensor
        """
        # calculate the forward pass
        if self.training:
          if x.ndim == 2:
            dim = 0
          elif x.ndim == 3:
            dim = (0,1)
          xmean = x.mean(dim, keepdim=True) # batch mean
          xvar = x.var(dim, keepdim=True) # batch variance
        else:
          xmean = self.running_mean
          xvar = self.running_var
        xhat = (x - xmean) / torch.sqrt(xvar + self.eps) # normalize to unit variance
        self.out = self.gamma * xhat + self.beta
        # update the buffers
        if self.training:
          with torch.no_grad():
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * xmean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * xvar
        return self.out
