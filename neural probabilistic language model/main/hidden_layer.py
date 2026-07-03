import torch

class HiddenLayer:

    def __init__(self, device='cpu', context_lenght=3, feature_dimension=2, num_neuron=100, generator=None):
        self.device = device
        self.generator = torch.Generator(device=self.device).manual_seed(2147483647) if generator == None else generator
        self.context_length = context_lenght
        self.feature_dimension = feature_dimension
        self.W = torch.randn(((self.context_length * self.feature_dimension), num_neuron), 
                             requires_grad=True, 
                             device=self.device, 
                             generator=self.generator)
        self.B = torch.randn(num_neuron, requires_grad=True, 
                             device=self.device, 
                             generator=self.generator)
    
    def __call__(self, embedding):
        output = torch.tanh(embedding.view(-1, (self.context_length * self.feature_dimension)) @ self.W + self.B)
        return output