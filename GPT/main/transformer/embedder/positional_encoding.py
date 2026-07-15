import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len, device='cpu'):
        super().__init__()
        self.d_model = torch.tensor(d_model, device=device)
        self.max_len = torch.tensor(max_len, device=device)
        self.positional_encoding = torch.zeros(max_len, d_model, requires_grad=False, device=device)
        position = torch.arange(0, max_len, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, device=device) * -(math.log(10000.0) / d_model))
        self.positional_encoding[:, 0::2] = torch.sin(position * div_term)
        self.positional_encoding[:, 1::2] = torch.cos(position * div_term)

    def forward(self, x):
        x = x + self.positional_encoding[:x.size(1), :]
        return x