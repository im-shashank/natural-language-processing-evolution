import torch.nn as nn
from encoder.encoder import Encoder
from decoder.decoder import Decoder

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, max_len, n_layers, n_heads, d_ff, device='cpu', dropout=0.1):
        super().__init__()
        self.encoder = Encoder(src_vocab_size, d_model, max_len, n_layers, n_heads, d_ff, device=device, dropout=dropout)
        
        # Decoder now needs vocab_size and max_len for its embeddings
        self.decoder = Decoder(tgt_vocab_size, d_model, max_len, n_heads, n_layers, device=device, dropout=dropout)
        
        self.linear = nn.Linear(d_model, tgt_vocab_size, device=device)
        # Note: PyTorch's CrossEntropyLoss expects raw logits, so you usually omit the Softmax here if training.

    def forward(self, x, y, src_pad_mask=None, tgt_mask=None, tgt_pad_mask=None):
        # 1. Encoder processes the source sequence `x`
        encoder_output = self.encoder(x, mask=src_pad_mask)
        
        # 2. Decoder processes the target sequence `y` AND the `encoder_output`
        decoder_output = self.decoder(y, encoder_output, tgt_mask, tgt_pad_mask, memory_pad_mask=src_pad_mask)
        
        # 3. Project to vocabulary size
        logits = self.linear(decoder_output)
        
        return logits


# if __name__ == "__main__":
#     # Hyperparameters
#     VOCAB_SIZE = 1000
#     MAX_LEN = 100
#     D_MODEL = 512
#     N_LAYERS = 6
#     N_HEADS = 8
#     D_FF = 2048
    
#     # Initialize the Encoder
#     encoder = Encoder(
#         vocab_size=VOCAB_SIZE, 
#         d_model=D_MODEL, 
#         max_len=MAX_LEN, 
#         n_layers=N_LAYERS, 
#         n_heads=N_HEADS, 
#         d_ff=D_FF
#     )
    
#     # Mock Batch: Batch Size = 2, Sequence Length = 10
#     dummy_input = torch.randint(0, VOCAB_SIZE, (2, 10)) 
    
#     # Forward Pass
#     output = encoder(dummy_input)
    
#     print(f"Input Shape:  {dummy_input.shape}")
#     print(f"Output Shape: {output.shape} (Batch Size, Seq Len, D_Model)")