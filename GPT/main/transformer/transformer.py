import torch
import torch.nn as nn

from transformer.encoder.encoder import Encoder
from transformer.decoder.decoder import Decoder

class Transformer(nn.Module):
    def __init__(self, src_vocab_size=0, tgt_vocab_size=0, d_model=0, max_len=0, n_layers=0, n_heads=0, d_ff=0, device='cpu', dropout=0.1):
        super().__init__()

        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size

        self.encoder = Encoder(src_vocab_size, d_model, max_len, n_layers, n_heads, d_ff, device=device, dropout=dropout)
        
        # Decoder now needs vocab_size and max_len for its embeddings
        self.decoder = Decoder(tgt_vocab_size, d_model, max_len, n_heads, n_layers, device=device, dropout=dropout)
        
        self.linear = nn.Linear(d_model, tgt_vocab_size, device=device)
        # Note: PyTorch's CrossEntropyLoss expects raw logits, so you usually omit the Softmax here if training.

    def forward(self, x, y, src_pad_mask=None, tgt_mask=None, tgt_pad_mask=None):
        mask_device = x.device
        src_len = x.size(1)
        tgt_len = y.size(1)

        # Causal masks everywhere so no position can read a future token.
        # Both the encoder and decoder see the same context in the LM setup, so the
        # encoder self-attention and the cross-attention must be causal too, otherwise
        # the next-token answer leaks straight into the prediction.
        src_mask = self._causal_mask(src_len, src_len, mask_device)
        if tgt_mask is None:
            tgt_mask = self._causal_mask(tgt_len, tgt_len, mask_device)
        memory_mask = self._causal_mask(tgt_len, src_len, mask_device)

        # 1. Encoder processes the source sequence `x`
        encoder_output = self.encoder(x, attn_mask=src_mask, key_padding_mask=src_pad_mask)
        
        # 2. Decoder processes the target sequence `y` AND the `encoder_output`
        decoder_output = self.decoder(y, encoder_output, tgt_mask=tgt_mask, tgt_pad_mask=tgt_pad_mask, memory_mask=memory_mask, memory_pad_mask=src_pad_mask)
        
        # 3. Project to vocabulary size
        logits = self.linear(decoder_output)
        
        return logits

    @staticmethod
    def _causal_mask(size_q, size_k, device):
        # Boolean mask where True marks positions that are NOT allowed to attend
        # (upper triangle above the diagonal = future tokens).
        return torch.triu(torch.ones(size_q, size_k, dtype=torch.bool, device=device), diagonal=1)

    def set_src_vocab_size(self, src_vocab_size:0):
        self.src_vocab_size = src_vocab_size

    def set_tgt_vocab_size(self, tgt_vocab_size:0):
        self.tgt_vocab_size = tgt_vocab_size
