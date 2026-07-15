import torch
import torch.nn as nn

class ScaledEmbedding(nn.Module):
    def __init__(self, d_model, max_len, device='cpu'):
        super().__init__()
        self.d_model = torch.tensor(d_model, dtype=torch.float32, device=device)
        self.max_len = torch.tensor(max_len, dtype=torch.float32, device=device)
        self.embedding = nn.Embedding(max_len, d_model, device=device)
        self.scale = torch.sqrt(self.d_model)

    def forward(self, x):
        x = self.embedding(x) * self.scale
        return x