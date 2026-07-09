# Simple Language Models

This repository contains implementations of three classic language models in Python using PyTorch:

1. **Bigram Language Model** - A simple n-gram model that predicts the next character based on the previous character
2. **Neural Probabilistic Language Model** - A more sophisticated neural network-based approach to language modeling
3. **WaveNet Language Model** - A state-of-the-art convolutional neural network for sequence modeling

## Project Structure

```
.
├── bi-gram language model/
│   ├── main/                    # Main implementation files
│   ├── resources/               # Training data
│   └── README.md                # Bigram model documentation
├── neural probabilistic language model/
│   ├── main/                    # Main implementation files
│   ├── resources/               # Training data
│   └── README.md                # Neural probabilistic model documentation
└── wave-net language model/
    ├── main/                    # Main implementation files
    ├── resources/               # Training data
    ├── Figure_1.png             # Training loss graph
    └── README.md                # WaveNet model documentation
```

## Bigram Language Model

The bigram language model is a simple character-level language model that predicts the next character based on the previous character. It uses a lookup table approach to store transition probabilities between characters.

### Features:
- Character-level language modeling
- Training from text data
- Weight saving and loading capabilities
- Generation of new text samples

### How to Run:
```bash
cd bi-gram\ language\ model/main
python main.py
```

## Neural Probabilistic Language Model

The neural probabilistic language model uses a neural network approach to predict the next character in a sequence. It consists of embedding, hidden, and softmax layers.

### Features:
- Neural network-based language modeling
- Embedding layer for character representations
- Hidden layer with tanh activation
- Softmax output layer for probability distribution
- Training from text data
- Dynamically adjusting learning rate depending upon if the loss increases or decreases.
- L2 Regularization (Weight Decay) during loss calculation

## WaveNet Language Model

The WaveNet language model implements a state-of-the-art convolutional neural network architecture for sequence modeling. It uses dilated convolutions and residual connections to efficiently capture long-range dependencies in text data.

### Features:
- Character-level language modeling
- Dilated convolutional layers for efficient sequence processing
- Batch normalization and dropout for stable training
- Residual connections for better gradient flow
- Training from text data with dynamic learning rate scheduling
- Loss visualization capabilities

### How to Run:
```bash
cd wave-net\ language\ model/main
python main.py
```
- Kaiming initialization of the hidden tanh layer.
- Batch normalization
- Weight manager class that saves and loads(once model is trained) the models weights.

### How to Run:
```bash
cd neural\ probabilistic\ language\ model/main
python main.py
```

## Requirements

Both models require Python 3 and PyTorch:

```bash
pip install torch
```

## Data

Training data is included in the `resources/` directories of each model. The default file used is `names.txt` which contains a list of names.

## License

This project is licensed under the MIT License.