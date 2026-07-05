import torch

class HiddenLayer:
    """
    Fully-connected hidden layer with tanh activation for the language model.

    This layer takes the flattened embedding vectors from the context window and
    projects them into a higher-dimensional hidden space. The tanh activation
    function introduces non-linearity, allowing the network to learn complex
    patterns in character sequences.

    The layer learns two sets of parameters:
      - W (weight matrix): Maps concatenated embeddings to hidden neurons.
      - B (bias vector): Per-neuron bias for shifting activations.

    Attributes:
        device (str): The device to create tensors on ('cpu', 'cuda', 'mps').
        generator (torch.Generator): Seeded random number generator for reproducibility.
        context_length (int): Number of characters in the input context window.
        feature_dimension (int): Dimensionality of each character's embedding vector.
        W (torch.Tensor): Weight matrix of shape (context_length * feature_dimension, num_neuron).
        B (torch.Tensor): Bias vector of shape (num_neuron,).
    """

    def __init__(self, device='cpu', context_lenght=3, feature_dimension=2, num_neuron=100, generator=None):
        """
        Initializes the hidden layer with randomly-initialized weights and biases.

        Args:
            device (str): The device to create tensors on. Defaults to 'cpu'.
            context_lenght (int): Number of characters in the input context window.
                Defaults to 3.
            feature_dimension (int): Dimensionality of each character's embedding vector.
                Defaults to 2.
            num_neuron (int): Number of hidden neurons in this layer. Defaults to 100.
            generator (torch.Generator, optional): Seeded random number generator for reproducibility.
                If None, creates a new generator with seed 2147483647.
        """
        self.device = device
        self.generator = torch.Generator(device=self.device).manual_seed(2147483647) if generator == None else generator
        self.context_length = context_lenght
        self.feature_dimension = feature_dimension
        self.tanh_gain = (5/3) # this is the standard gain for a tanh non-linearity
        self.kaiming_init = ( self.tanh_gain / (((self.context_length * self.feature_dimension)) ** 0.5))
        self.W = (torch.randn(((self.context_length * self.feature_dimension), num_neuron), 
                             device=self.device, 
                             generator=self.generator) * self.kaiming_init).requires_grad_(True)
        self.B = (torch.randn(num_neuron, 
                             device=self.device, 
                             generator=self.generator) * 0.01).requires_grad_(True)
    
    def __call__(self, embedding):
        """
        Performs a forward pass through the hidden layer.

        Flattens the embedding tensor and applies a linear transformation followed
        by a tanh activation function:  output = tanh(embedding_flat @ W + B)

        Args:
            embedding (torch.Tensor): Embedding tensor of shape
                (batch_size, context_length, feature_dimension).

        Returns:
            torch.Tensor: Activated hidden representation of shape (batch_size, num_neuron),
                where values are in the range [-1, 1] due to tanh activation.
        """
        output = torch.tanh(embedding.view(-1, (self.context_length * self.feature_dimension)) @ self.W + self.B)
        return output