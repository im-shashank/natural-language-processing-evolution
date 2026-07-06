import torch

class BatchNormalization:

    def __init__(self, device='cpu', num_neuron=100):
        self.batch_normalization_gain = torch.ones((1, num_neuron), device=device, requires_grad=True)
        self.batch_normalization_bias = torch.zeros((1, num_neuron), device=device, requires_grad=True)
        self.epsilon = 1e-5

        self.batch_normalization_mean_running = torch.zeros((1, num_neuron), device=device, requires_grad=False)
        self.batch_normalization_variance_running = torch.ones((1, num_neuron), device=device, requires_grad=False)

    def __call__(self, embedding_concate, training=True):
        if training:
            embedding_concate_mean = embedding_concate.mean(0, keepdim=True)
            embedding_concate_std_deviation = embedding_concate.std(0, keepdim=True)
            # First normalize
            x_hat = (embedding_concate - embedding_concate_mean) / (embedding_concate_std_deviation + self.epsilon)
            with torch.no_grad():
                self.batch_normalization_mean_running = 0.999 * self.batch_normalization_mean_running + 0.001 * embedding_concate_mean
                self.batch_normalization_variance_running = 0.999* self.batch_normalization_variance_running + 0.001 * embedding_concate_std_deviation
        else:
            # Use the running stats during evaluation/sampling
            x_hat = (embedding_concate - self.batch_normalization_mean_running) / (self.batch_normalization_variance_running + self.epsilon)
        
        # Then scale and shift
        batch_normalized = (self.batch_normalization_gain * x_hat) + self.batch_normalization_bias
        return batch_normalized