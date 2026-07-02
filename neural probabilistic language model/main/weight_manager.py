import torch
import os

#TODO: Needs correct implementation
class WeightManager:
    """
    A class to manage saving and loading weights for the Bigram Language Model.
    """
    def __init__(self, filepath="bigram_weights.pt"):
        self.filepath = filepath

    def save_weights(self, model):
        """
        Saves the neural network's weight matrix to a file.
        """
        # We save the underlying data of the weights tensor
        torch.save(model.parameters.data, self.filepath)
        print(f"Weights successfully saved to '{self.filepath}'.")

    def load_weights(self, model, device):
        """
        Loads the weights from a file and populates the model's weight matrix.
        Returns True if successful, False if the file doesn't exist.
        """
        if os.path.exists(self.filepath):
            # map_location ensures the weights load onto the correct hardware (CPU/MPS/CUDA)
            loaded_weights = torch.load(self.filepath, map_location=device)
            
            # Update the underlying data of the neuron's weight matrix
            model.parameters.data = loaded_weights
            print(f"Weights successfully loaded from '{self.filepath}'.")
            return True
        else:
            print(f"No saved weights found at '{self.filepath}'.")
            return False