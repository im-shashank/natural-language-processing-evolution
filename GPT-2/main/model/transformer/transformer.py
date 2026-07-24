import torch
import torch.nn as nn

from model.transformer.decoder.decoder import Decoder

class Transformer(nn.Module):
    def __init__(self, vocab_size=0, d_model=0, max_len=0, n_layers=0, n_heads=0, d_ff=0, device='cpu', dropout=0.1):
        super().__init__()

        self.vocab_size = vocab_size

        self.decoder = Decoder(vocab_size, d_model, max_len, n_heads, n_layers, d_ff=d_ff, device=device, dropout=dropout)
        
        # GPT-2 specific: Final LayerNorm before the linear projection
        self.ln_f = nn.LayerNorm(d_model, device=device)
        self.linear = nn.Linear(d_model, vocab_size, device=device)

    def forward(self, x, tgt_mask=None, tgt_pad_mask=None):
        mask_device = x.device
        seq_len = x.size(1)

        # Causal mask so no position can read a future token.
        if tgt_mask is None:
            tgt_mask = self._causal_mask(seq_len, seq_len, mask_device)
        
        decoder_output = self.decoder(x, tgt_mask=tgt_mask, tgt_pad_mask=tgt_pad_mask)
        decoder_output = self.ln_f(decoder_output)
        
        logits = self.linear(decoder_output)
        
        return logits

    @staticmethod
    def _causal_mask(size_q, size_k, device):
        # Boolean mask where True marks positions that are NOT allowed to attend
        # (upper triangle above the diagonal = future tokens).
        return torch.triu(torch.ones(size_q, size_k, dtype=torch.bool, device=device), diagonal=1)

    def set_vocab_size(self, vocab_size):
        self.vocab_size = vocab_size
