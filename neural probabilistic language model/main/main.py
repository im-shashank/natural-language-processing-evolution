"""
Main entry point for the Neural Probabilistic Language Model.

This module initializes and runs the full training pipeline for a character-level
neural language model inspired by Karpathy's "Neural Network: Zero to Hero" series.
The model learns to generate English names from a dataset of ~32K names.

This model also has two additional features that are completely my own ideas.
    1. A learning rate decay function in neural_probabilistic_language_model.py file.
    2. A model weights manager script that saves and loads the weights of model after it
       is trained. 

Pipeline:
    1. Split the names dataset into train / validation / test sets (80/10/10).
    2. Train the model on the training set using backpropagation.
    3. Evaluate the trained model on the validation and test sets.
    4. Sample new names from the trained model to demonstrate generation capability.

Architecture:
    Input (character indices) → Embedding Layer → Hidden Layer (tanh) → Softmax Layer → Output probabilities

Usage:
    Run directly with: python main.py
"""

from neural_probabilistic_language_model import *
from weight_manager import *

# Initialize the model and weight manager
npblm = NeuralProbabilisticLanguageModel()

#Weight manager needs to be implemented to save and load weights
weight_manager = ModelWeightManager(filepath="neural probabilistic language model/neural_probabilistic_language_model_weights.pt")

# Attempt to load existing weights
weights_exist = weight_manager.load(npblm, device)

if not weights_exist:
    print("Training model from scratch...")
    npblm(train_model=True, validate_model=True, test_model=True)

    # Save the weights so we don't have to train next time
    weight_manager.save(npblm, 
                        npblm.learning_iteration,
                        npblm.training_loss,
                        npblm.validation_loss,
                        npblm.test_loss,
                        npblm.learning_rate)

#Sample from the model
print("output sampled from model is:")
npblm.sample_from_model()