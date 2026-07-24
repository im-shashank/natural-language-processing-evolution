import math
import torch.nn as nn
import torch

class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, n_layers, d_ff=None, device='cpu', dropout=0.1):
        super().__init__()
        self.n_layers = n_layers
        if d_ff is None:
            d_ff = d_model * 4
        self.masked_mha = nn.MultiheadAttention(d_model, n_heads, batch_first=True, device=device)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff, device=device),
            nn.GELU(), # GPT-2 typically uses GELU
            nn.Linear(d_ff, d_model, device=device)
        )
        self.layernorm1 = nn.LayerNorm(d_model, device=device)
        self.layernorm2 = nn.LayerNorm(d_model, device=device)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        # Apply GPT-2 specific residual initialization scaling
        self._scale_residual_weights()
        
    def _scale_residual_weights(self):
        """Scales the weights of residual projections at initialization."""
        # N is the total number of residual layers in the network.
        # Since there are 2 residual connections per block, N = n_layers * 2
        N = self.n_layers * 2
        scaling_factor = 1 / math.sqrt(N)
        
        # Scale the weights of the Multi-Head Attention output projection
        with torch.no_grad():
            if hasattr(self.masked_mha, 'out_proj'):
                self.masked_mha.out_proj.weight.data *= scaling_factor
            
            # Scale the weights of the FFN output projection (the final linear layer in the FFN)
            self.ffn[2].weight.data *= scaling_factor
    
    def forward(self, x, tgt_mask=None, tgt_pad_mask=None):
        # 1. Masked Self-Attention with Pre-LayerNorm
        residual = x
        x_norm = self.layernorm1(x)
        attn_output, _ = self.masked_mha(x_norm, x_norm, x_norm, attn_mask=tgt_mask, key_padding_mask=tgt_pad_mask)
        x = residual + self.dropout1(attn_output)

        # 2. Feed-Forward with Pre-LayerNorm
        residual = x
        x_norm = self.layernorm2(x)
        ffn_output = self.ffn(x_norm)
        x = residual + self.dropout2(ffn_output)
        
        return x