import torch
import torch.nn as nn
from transformer.embedder.scaled_embedding import ScaledEmbedding
from transformer.embedder.positional_encoding import PositionalEncoding
from transformer.decoder.transformer_decoder_layer import TransformerDecoderLayer

class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, max_len, n_heads, n_layers, device='cpu', dropout=0.1):
        super().__init__()
        # 1. Add Embeddings for the target sequence
        self.embedding = ScaledEmbedding(d_model, vocab_size, device=device)
        self.pos_encoding = PositionalEncoding(d_model, max_len, device=device)
        self.dropout = nn.Dropout(dropout)
        
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, n_heads, device=device, dropout=dropout) 
            for _ in range(n_layers)
        ])

    def forward(self, y, encoder_output, tgt_mask=None, tgt_pad_mask=None, memory_mask=None, memory_pad_mask=None):
        # Embed the target sequence
        y = self.embedding(y)
        y = self.pos_encoding(y)
        y = self.dropout(y)

        # Pass y AND encoder_output through each decoder layer sequentially
        for layer in self.layers:
            y = layer(y, encoder_output, tgt_mask, tgt_pad_mask, memory_mask, memory_pad_mask)

        return y