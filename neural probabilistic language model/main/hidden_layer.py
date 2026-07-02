import torch

class HiddenLayer:

    def __init__(self, device='cpu', context_lenght=3, feature_dimension=2, num_neuron=100):
        self.device = device
        self.context_length = context_lenght
        self.feature_dimension = feature_dimension
        self.W = torch.randn(((self.context_length * self.feature_dimension), num_neuron), requires_grad=True, device=self.device)
        self.B = torch.randn(num_neuron, requires_grad=True, device=self.device)
    
    def __call__(self, embedding):
        output = torch.tanh(embedding.view(-1, (self.context_length * self.feature_dimension)) @ self.W + self.B)
        return output