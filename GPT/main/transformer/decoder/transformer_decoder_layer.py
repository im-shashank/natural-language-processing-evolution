import torch.nn as nn

class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, device='cpu', dropout=0.1):
        super().__init__()
        self.masked_mha = nn.MultiheadAttention(d_model, n_heads, batch_first=True, device=device)
        self.crossed_mha = nn.MultiheadAttention(d_model, n_heads, batch_first=True, device=device)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4, device=device),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model, device=device)
        )
        self.layernorm1 = nn.LayerNorm(d_model, device=device)
        self.layernorm2 = nn.LayerNorm(d_model, device=device)
        self.layernorm3 = nn.LayerNorm(d_model, device=device)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
    
    def forward(self, x, encoder_output, tgt_mask=None, tgt_pad_mask=None, memory_pad_mask=None):

        # 1. Masked Self-Attention: Uses tgt_mask (causal look-ahead) and tgt_pad_mask
        attn_output, _ = self.masked_mha(x, x, x, attn_mask=tgt_mask, key_padding_mask=tgt_pad_mask)
        x = self.layernorm1(x + self.dropout1(attn_output))

        # 2. Cross Attention: Uses memory_pad_mask (encoder's padding mask)
        attn_output, _ = self.crossed_mha(x, encoder_output, encoder_output, key_padding_mask=memory_pad_mask)
        x = self.layernorm2(x + self.dropout2(attn_output))
        
        # 3. Feed-Forward
        ffn_output = self.ffn(x)
        x = self.layernorm3(x + self.dropout3(ffn_output))
        
        return x