import torch
import torch.nn.functional as F
from tokenizer import *


class TrainingSet:
    """
    A class for preparing training data for the language model.
    
    This class handles loading the dataset and creating input-output pairs
    suitable for training a neural network.
    
    Attributes:
        words (list): List of words from the dataset.
        tokenizer (Tokenizer): Instance of Tokenizer for character mapping.
    """

    def __init__(self, device):
        """
        Initializes the TrainingSet with the dataset.
        
        Loads words from the names.txt file and creates a tokenizer instance.
        """
        self.words = open('bi-gram language model/resources/names.txt', 'r').read().splitlines()
        self.tokenizer = Tokenizer(self.words)
        self.device = device

    def create_training_set_for_neural_net(self):
        """
        Creates training data for the neural network.
        
        This method processes all words in the dataset to create input-output pairs
        where inputs are current characters and outputs are next characters.
        
        Returns:
            list: A list containing [xenc, ys] where xenc is one-hot encoded inputs
                  and ys are the corresponding target indices.
        """
        xs, ys = [], []
        for w in self.words:
            chs = [self.tokenizer.special_char] + list(w) + [self.tokenizer.special_char]
            for ch1, ch2 in zip(chs, chs[1:]):
                ix1 = self.tokenizer.stoi[ch1]
                ix2 = self.tokenizer.stoi[ch2]
                xs.append(ix1)
                ys.append(ix2)
        
        xs = torch.tensor(xs, device=self.device) #Move to device (preferably GPU) if not then CPU
        ys = torch.tensor(ys, device=self.device) #Move to device (preferably GPU) if not then CPU
        num_elements = xs.nelement()
        xenc = F.one_hot(xs, num_classes=27).float() #PyTorch will automatically move xenc to GPU

        return [xenc, ys, num_elements]