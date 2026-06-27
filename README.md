# Simple Bigram Language Model

Hey there! Welcome to this simple, character-level bigram language model built from scratch using PyTorch. 

If you've ever wondered how neural networks learn to generate text, this project is a perfect starting point. Instead of generating whole sentences like ChatGPT, this model learns to generate novel words, character by character, by studying the statistical patterns in a dataset of names.

## What does it do?
At its core, this model asks a very simple question: *"Given the current character, what is the most likely next character?"* It uses a 1-layer neural network (essentially a learnable lookup table) to figure out the probabilities of character transitions. For example, if the current letter is `q`, the network learns that the next letter is almost certainly going to be `u`. 

## Project Structure
The code is organized to be highly modular and easy to digest:

* **`main.py`**: The entry point. It orchestrates the loading/training of the model and prints out the generated words.
* **`bigram_language_model.py`**: Contains the main training loop, loss calculation (Negative Log-Likelihood), and the text generation logic.
* **`bigram_neuron.py`**: The "brain" of the operation. It holds our `27x27` weight matrix (representing 26 alphabet letters + 1 special start/end token) and applies a softmax activation to convert raw log counts into clean probabilities.
* **`training_set.py`**: Handles reading the raw text file and formatting it into one-hot encoded input tensors and target labels for PyTorch.
* **`tokenizer.py`**: Responsible for creating the vocabulary. It maps characters to integers (`stoi`) and integers back to characters (`itos`).
* **`weight_manager.py`**: A handy utility that saves the trained weights to `bigram_weights.pt` so you don't have to retrain the model every single time you run the script!

## Key Features
* **Hardware Acceleration**: Automatically detects and uses Apple's Metal Performance Shaders (MPS) if you're on a Mac with Apple Silicon, falling back to CPU if not available.
* **Smart Training Loop**: Includes an early stopping mechanism that safely halts training once the loss plateaus around the mathematical limit for a bigram model (~2.48).
* **Weight Persistence**: Saves and loads weights automatically.

## How to Run It

1. **Prerequisites**: Make sure you have PyTorch installed (`pip install torch`).
2. **Dataset**: Ensure you have a text file containing your dataset located at `resources/names.txt` (one word per line).
3. **Execute**:

## What to expect when you run it:
The first time you run the script, it won't find a bigram_weights.pt file, so it will train the network from scratch. You'll see the loss smoothly decrease in the console until it hits its theoretical plateau. After that, it will save the weights and instantly spit out 20 brand-new, randomly generated words!

On subsequent runs, it will bypass training entirely, load the saved weights, and jump straight into generating text.

You can change the manual seed of torch.Generator in bigram_language_model.py (line 100) to generate new output.


## My output:

```text
current loss is : 2.4844515323638916 | current leaning rate is: 0.1 | current iterator count is: 79900
current loss is : 2.484441041946411 | current leaning rate is: 0.1 | current iterator count is: 80000
current loss is : 2.484431028366089 | current leaning rate is: 0.1 | current iterator count is: 80100
current loss is : 2.4844212532043457 | current leaning rate is: 0.1 | current iterator count is: 80200
Training converged early. Stopping.
Weights successfully saved to 'bigram_weights.pt'.

Sampling from model:
['ten', 'has', 'daleder', 'dvosttanderide', 'dur', 'wi', 'giadenda', 'parire', 't', 'katyanadele', 'ay', 'ra', 'mahlynna', 'regheee', 'cadaren', 'morili', 'ga', 'ar', 'sinway', 'maraxrrecaycee']
```

**Happy generating!**