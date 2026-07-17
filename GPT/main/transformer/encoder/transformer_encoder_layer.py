import torch
import torch.nn as nn

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, device='cpu', dropout=0.1):
        super().__init__()
        self.mha = nn.MultiheadAttention(d_model, n_heads, batch_first=True, device=device)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4, device=device),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model, device=device)
        )
        self.layernorm1 = nn.LayerNorm(d_model, device=device)
        self.layernorm2 = nn.LayerNorm(d_model, device=device)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, x, attn_mask=None, key_padding_mask=None):
        # 1. Self-Attention + Residual Connection (attn_mask = causal look-ahead mask)
        attn_output, _ = self.mha(x, x, x, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        x = self.layernorm1(x + self.dropout1(attn_output))
        
        # 2. Feed-Forward + Residual Connection
        ffn_output = self.ffn(x)
        x = self.layernorm2(x + self.dropout2(ffn_output))
        
        return x