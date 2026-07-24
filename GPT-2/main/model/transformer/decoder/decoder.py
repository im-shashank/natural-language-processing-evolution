import torch
import torch.nn as nn
from model.transformer.embedder.scaled_embedding import ScaledEmbedding
from model.transformer.embedder.positional_encoding import PositionalEncoding
from model.transformer.decoder.transformer_decoder_layer import TransformerDecoderLayer

class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, max_len, n_heads, n_layers, d_ff=None, device='cpu', dropout=0.1):
        super().__init__()
        # 1. Add Embeddings for the target sequence
        self.embedding = ScaledEmbedding(d_model, vocab_size, device=device)
        self.pos_encoding = PositionalEncoding(d_model, max_len, device=device)
        self.dropout = nn.Dropout(dropout)
        
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, n_heads, n_layers, d_ff=d_ff, device=device, dropout=dropout) 
            for _ in range(n_layers)
        ])

    def forward(self, x, tgt_mask=None, tgt_pad_mask=None):
        # Embed the sequence
        x = self.embedding(x)
        x = self.pos_encoding(x)
        x = self.dropout(x)

        # Pass x through each decoder layer sequentially
        for layer in self.layers:
            x = layer(x, tgt_mask, tgt_pad_mask)

        return x