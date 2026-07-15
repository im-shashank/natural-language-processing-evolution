import torch
import sys
sys.path.append("main/transformer")

from transformer import Transformer

# Check if MPS (Metal Performance Shaders) is available for Apple Silicon
device = torch.device("cuda" if torch.cuda.is_available()
                      else "mps" if torch.backends.mps.is_available()
                         else "cpu")

def test_transformer_output_shape():
    """Test that the Transformer produces the correct output shape."""
    # Hyperparameters
    SRC_VOCAB_SIZE = 1000
    TGT_VOCAB_SIZE = 1500
    MAX_LEN = 100
    D_MODEL = 512
    N_LAYERS = 6
    N_HEADS = 8
    D_FF = 2048
    BATCH_SIZE = 2
    SRC_SEQ_LEN = 10
    TGT_SEQ_LEN = 12

    model = Transformer(
        src_vocab_size=SRC_VOCAB_SIZE,
        tgt_vocab_size=TGT_VOCAB_SIZE,
        d_model=D_MODEL,
        max_len=MAX_LEN,
        n_layers=N_LAYERS,
        n_heads=N_HEADS,
        d_ff=D_FF,
        device=device
    )

    src = torch.randint(0, SRC_VOCAB_SIZE, (BATCH_SIZE, SRC_SEQ_LEN), device=device)
    tgt = torch.randint(0, TGT_VOCAB_SIZE, (BATCH_SIZE, TGT_SEQ_LEN), device=device)

    logits = model(src, tgt)

    assert logits.shape == (BATCH_SIZE, TGT_SEQ_LEN, TGT_VOCAB_SIZE), \
        f"Expected shape {(BATCH_SIZE, TGT_SEQ_LEN, TGT_VOCAB_SIZE)}, got {logits.shape}"


def test_transformer_with_masks():
    """Test that the Transformer works with padding masks."""
    SRC_VOCAB_SIZE = 500
    TGT_VOCAB_SIZE = 500
    MAX_LEN = 50
    D_MODEL = 256
    N_LAYERS = 2
    N_HEADS = 4
    D_FF = 1024
    BATCH_SIZE = 4
    SRC_SEQ_LEN = 8
    TGT_SEQ_LEN = 6

    model = Transformer(
        src_vocab_size=SRC_VOCAB_SIZE,
        tgt_vocab_size=TGT_VOCAB_SIZE,
        d_model=D_MODEL,
        max_len=MAX_LEN,
        n_layers=N_LAYERS,
        n_heads=N_HEADS,
        d_ff=D_FF,
        device=device
    )

    src = torch.randint(0, SRC_VOCAB_SIZE, (BATCH_SIZE, SRC_SEQ_LEN), device=device)
    tgt = torch.randint(0, TGT_VOCAB_SIZE, (BATCH_SIZE, TGT_SEQ_LEN), device=device)

    # Causal mask for the decoder (upper triangular)
    tgt_mask = torch.triu(torch.ones(TGT_SEQ_LEN, TGT_SEQ_LEN, device=device), diagonal=1).bool()

    logits = model(src, tgt, tgt_mask=tgt_mask)

    assert logits.shape == (BATCH_SIZE, TGT_SEQ_LEN, TGT_VOCAB_SIZE)


def test_transformer_gradient_flow():
    """Test that gradients flow back through the model."""
    model = Transformer(
        src_vocab_size=100,
        tgt_vocab_size=100,
        d_model=64,
        max_len=20,
        n_layers=1,
        n_heads=2,
        d_ff=128,
        device=device
    )

    src = torch.randint(0, 100, (1, 5), device=device)
    tgt = torch.randint(0, 100, (1, 5), device=device)

    logits = model(src, tgt)
    loss = logits.sum()
    loss.backward()

    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"No gradient for {name}"


if __name__ == "__main__":
    print(f"Using device: {device}")

    test_transformer_output_shape()
    print("✓ test_transformer_output_shape passed")

    test_transformer_with_masks()
    print("✓ test_transformer_with_masks passed")

    test_transformer_gradient_flow()
    print("✓ test_transformer_gradient_flow passed")

    print("\nAll tests passed!")