import torch
from bigram_language_model import *
from weight_manager import WeightManager

# Initialize the model and the weight manager
blm = BigramLanguageModel()
weight_manager = WeightManager(filepath="bi-gram language model/bigram_weights.pt")

# Attempt to load existing weights
weights_exist = weight_manager.load_weights(blm, device)

if not weights_exist:
    print("Training model from scratch...")
    blm.train_model()
    
    # Save the weights so we don't have to train next time
    weight_manager.save_weights(blm)

# Test the model using the generation function we created earlier
print("\nSampling from model:")
sampled_words = blm.generate_words(num_words=20)
print(sampled_words)