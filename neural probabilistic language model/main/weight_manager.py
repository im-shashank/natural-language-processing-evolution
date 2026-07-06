import torch
import os


class ModelWeightManager:
    """
    Manages saving and loading all parameters for the Neural Probabilistic Language Model.
    
    Saves/loads the complete model state atomically as a single checkpoint file,
    following PyTorch's state_dict convention. The checkpoint includes:
      - All 5 parameter tensors (embedding_C, hidden_W, hidden_B, softmax_W, softmax_B)
      - Training metadata (iteration, loss, learning_rate) for reproducibility
    
    Usage:
        manager = ModelWeightManager(filepath="model_checkpoint.pt")
        
        # Save during training
        manager.save(model, iteration=1000, loss=2.5, learning_rate=-0.09)
        
        # Load before testing
        success = manager.load(model, device=torch.device("cpu"))
    """

    def __init__(self, filepath="model_checkpoint.pt"):
        self.filepath = filepath

    def save(self, model, iteration=None, training_loss=None, validation_loss=None, test_loss=None, learning_rate=None):
        """
        Saves all model parameters and optional training metadata to a single file.
        
        Args:
            model: The NeuralProbabilisticLanguageModel instance.
            iteration: Current training iteration number (stored as metadata).
            loss: Current loss value (stored as metadata).
            learning_rate: Current learning rate (stored as metadata).
        """
        checkpoint = {
            # All 5 parameter tensors
            "embedding_C": model.embedding_layer.C.data,
            "hidden_W": model.hidden_layer.W.data,
            # "hidden_B": model.hidden_layer.B.data,
            "batch_normalization_gain": model.batch_normalization.batch_normalization_gain,
            "batch_normalization_bias": model.batch_normalization.batch_normalization_bias,
            "softmax_W": model.softmax_layer.W.data,
            "softmax_B": model.softmax_layer.B.data,
            # Training metadata for reproducibility
            "metadata": {
                "iteration": iteration,
                "training_loss": training_loss,
                "validation_loss": validation_loss,
                "test_loss": test_loss,
                "learning_rate": learning_rate,
            }
        }
        torch.save(checkpoint, self.filepath)
        print(f"Model checkpoint saved to '{self.filepath}'.")

    def load(self, model, device):
        """
        Loads all model parameters from a checkpoint file.
        
        Args:
            model: The NeuralProbabilisticLanguageModel instance to populate.
            device: The device to load tensors onto (cpu, cuda, mps).
            
        Returns:
            dict: The metadata from the checkpoint (iteration, loss, learning_rate),
                  or None if loading failed.
        """
        if not os.path.exists(self.filepath):
            print(f"No checkpoint found at '{self.filepath}'.")
            return None

        checkpoint = torch.load(self.filepath, map_location=device, weights_only=True)

        # Restore each parameter tensor
        model.embedding_layer.C.data = checkpoint["embedding_C"]
        model.hidden_layer.W.data = checkpoint["hidden_W"]
        # model.hidden_layer.B.data = checkpoint["hidden_B"]
        model.batch_normalization.batch_normalization_gain = checkpoint["batch_normalization_gain"]
        model.batch_normalization.batch_normalization_bias = checkpoint["batch_normalization_bias"]
        model.softmax_layer.W.data = checkpoint["softmax_W"]
        model.softmax_layer.B.data = checkpoint["softmax_B"]

        metadata = checkpoint.get("metadata", {})
        print(f"\nModel checkpoint loaded from:\n'{self.filepath}'\n")
        if metadata:
            print(f"\nCheckpoint info:\niteration={metadata.get('iteration')}\n"
                  f"training loss={metadata.get('training_loss')}\nvalidation loss={metadata.get('validation_loss')}\ntest loss={metadata.get('test_loss')}\nlearning rate={-metadata.get('learning_rate')}\n")
        return metadata