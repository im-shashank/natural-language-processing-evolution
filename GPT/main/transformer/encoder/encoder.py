import torch
import torch.nn as nn

from transformer.embedder.scaled_embedding import ScaledEmbedding
from transformer.embedder.positional_encoding import PositionalEncoding
from transformer.encoder.transformer_encoder_layer import TransformerEncoderLayer

# Assuming ScaledEmbedding and PositionalEmbedding are imported or defined above
class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, max_len, n_layers, n_heads, d_ff, device='cpu', dropout=0.1):
        super().__init__()
        # 1. Embeddings (Combining your Scaled and Positional modules)
        self.embedding = ScaledEmbedding(d_model, vocab_size, device=device) # max_len in your code represents vocab_size for nn.Embedding
        self.pos_encoding = PositionalEncoding(d_model, max_len, device=device)
        self.dropout = nn.Dropout(dropout)
        
        # 2. Stack of Encoder Layers
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, n_heads, d_ff, device=device, dropout=dropout) 
            for _ in range(n_layers)
        ])

    def forward(self, x, attn_mask=None, key_padding_mask=None):
        # Pass through embedding and scale it
        x = self.embedding(x)
        # Add positional embedding
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        # Pass through each encoder layer sequentially
        for layer in self.layers:
            x = layer(x, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
            
        return x