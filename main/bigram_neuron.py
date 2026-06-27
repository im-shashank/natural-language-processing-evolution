import torch

class BiGramNeuron:
    """
    A class representing a bigram neuron for character-level language modeling.
    
    This neuron acts as a learnable lookup table that predicts the probability
    of the next character given the current character using matrix multiplication.
    
    Attributes:
        W (torch.Tensor): Weight matrix of shape (27, 27) for character transitions.
        B (torch.Tensor): Bias term (currently set to 0.0).
    """

    def __init__(self, device):
        """
        Initializes the BiGramNeuron with random weights.
        
        The weights are initialized using a random generator with a fixed seed
        for reproducibility. The bias is currently set to 0.0.
        """
        g = torch.Generator(device=device).manual_seed(2147483647)
        self.W = torch.randn((27,27), generator=g, device=device, requires_grad=True) # Move to device (preferable GPU) if available
        
        self.B = 0.0
    
    #forward pass of the neuron
    def __call__(self, xenc):
        """
        Performs the forward pass through the neuron.
        
        Args:
            xenc (torch.Tensor): One-hot encoded input tensor of shape (n, 27).
            
        Returns:
            torch.Tensor: Probability distribution over next characters.
        """
        logits = xenc @ self.W # log counts
        out = self._softmax(logits)
        return out
    
    # This is the activation function, softmax (google "softmax activation function for more information")
    def _softmax(self, logits):
        """
        Applies the softmax activation function to logits.
        
        Args:
            logits (torch.Tensor): Input tensor containing unnormalized scores.
            
        Returns:
            torch.Tensor: Probability distribution after applying softmax.
        """
        counts = logits.exp()
        probs = counts / counts.sum(1, keepdims=True)
        return probs