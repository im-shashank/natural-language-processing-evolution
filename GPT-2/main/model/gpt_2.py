import os
import torch

from tokenizer.tokenizer import Tokenizer

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class GPT2:
    """A GPT-2 language model implementation.
    
    This class orchestrates the tokenizer, model training, validation, and loss calculation.
    """
    
    def __init__(self):
        """Initializes the GPT-2 model hyperparameters and tokenizer."""
        self.hyperparamters = {
            "compression_ratio": 1.5
        }

        self.tokenizer = Tokenizer(self.hyperparamters["compression_ratio"])
        pass

    def __call__(self):
        text1 = "Ｕｎｉｃｏｄｅ! 🅤🅝🅘🅒🅞🅓🅔‽ 🇺‌🇳‌🇮‌🇨‌🇴‌🇩‌🇪! 😄 The very name strikes fear and awe into the hearts of programmers worldwide. We all know we ought to “support Unicode” in our software (whatever that means—like using wchar_t for all the strings, right?). But Unicode can be abstruse, and diving into the thousand-page Unicode Standard plus its dozens of supplementary annexes, reports, and notes can be more than a little intimidating. I don’t blame programmers for still finding the whole thing mysterious, even 30 years after Unicode’s inception."
        text2 = "Hello world"
        text3 = "My name is shashank bhardwaj"

        self.tokenizer.train(text2)
        
        encoded2 = self.tokenizer.encode(text2)
        encoded3 = self.tokenizer.encode(text3)

        
        decoded2 = self.tokenizer.decode(encoded=encoded2)
        print(decoded2)
        decoded3 = self.tokenizer.decode(encoded=encoded3)
        print(decoded3)

    def validate_model(self):
        """Validates the model against a validation dataset."""
        pass

    def train_model(
        self, 
        training_src_dataset, 
        training_target_dataset, 
        validation_src_dataset, 
        validation_target_dataset
    ):
        """Trains the GPT-2 model.
        
        Args:
            training_src_dataset: The dataset used for training source inputs.
            training_target_dataset: The dataset used for training target labels.
            validation_src_dataset: The dataset used for validation source inputs.
            validation_target_dataset: The dataset used for validation target labels.
        """
        pass

    def calculate_loss(self, logits):
        """Calculates the loss for the model predictions.
        
        Args:
            logits (Tensor): The raw model outputs (logits).
            
        Returns:
            Tensor: The calculated loss value.
        """
        pass