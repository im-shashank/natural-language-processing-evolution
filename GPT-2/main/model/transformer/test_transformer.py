import os
import sys
import torch

current_dir = os.path.dirname(os.path.abspath(__file__))
main_dir = os.path.abspath(os.path.join(current_dir, "..", ".."))
if main_dir not in sys.path:
    sys.path.insert(0, main_dir)

from model.transformer.transformer import Transformer

# Check if MPS (Metal Performance Shaders) is available for Apple Silicon
device = torch.device("cuda" if torch.cuda.is_available()
                      else "mps" if torch.backends.mps.is_available()
                         else "cpu")

def test_transformer_output_shape():
    """Test that the Transformer produces the correct output shape."""
    # Hyperparameters
    VOCAB_SIZE = 1500
    MAX_LEN = 100
    D_MODEL = 512
    N_LAYERS = 6
    N_HEADS = 8
    D_FF = 2048
    BATCH_SIZE = 2
    SEQ_LEN = 12

    model = Transformer(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        max_len=MAX_LEN,
        n_layers=N_LAYERS,
        n_heads=N_HEADS,
        d_ff=D_FF,
        device=device
    )

    x = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN), device=device)

    logits = model(x)

    assert logits.shape == (BATCH_SIZE, SEQ_LEN, VOCAB_SIZE), \
        f"Expected shape {(BATCH_SIZE, SEQ_LEN, VOCAB_SIZE)}, got {logits.shape}"


def test_transformer_with_masks():
    """Test that the Transformer works with padding masks."""
    VOCAB_SIZE = 500
    MAX_LEN = 50
    D_MODEL = 256
    N_LAYERS = 2
    N_HEADS = 4
    D_FF = 1024
    BATCH_SIZE = 4
    SEQ_LEN = 6

    model = Transformer(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        max_len=MAX_LEN,
        n_layers=N_LAYERS,
        n_heads=N_HEADS,
        d_ff=D_FF,
        device=device
    )

    x = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN), device=device)

    # Causal mask for the decoder (upper triangular)
    tgt_mask = torch.triu(torch.ones(SEQ_LEN, SEQ_LEN, device=device), diagonal=1).bool()

    logits = model(x, tgt_mask=tgt_mask)

    assert logits.shape == (BATCH_SIZE, SEQ_LEN, VOCAB_SIZE)


def test_transformer_gradient_flow():
    """Test that gradients flow back through the model."""
    model = Transformer(
        vocab_size=100,
        d_model=64,
        max_len=20,
        n_layers=1,
        n_heads=2,
        d_ff=128,
        device=device
    )

    x = torch.randint(0, 100, (1, 5), device=device)

    logits = model(x)
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