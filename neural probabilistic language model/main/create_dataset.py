import torch
import torch.nn.functional as F
from tokenizer import *

class CreateDataset:
    """
    Creates training/validation/test datasets from a list of words (names).

    For each word, this class builds sliding-window input-target pairs where the
    input is a fixed-length context of character indices and the target is the
    next character index. A special token '.' (index 0) marks the end of a word.

    Example:
        Given word "abc" with context_length=3:
          [0, 0, 0] -> 'a'
          [0, 0, 'a'] -> 'b'
          [0, 'a', 'b'] -> 'c'
          ['a', 'b', 'c'] -> '.'

    Attributes:
        device (str): The device to create tensors on ('cpu', 'cuda', 'mps').
        generator (torch.Generator): Seeded random number generator for reproducibility.
        words (list): List of words to build the dataset from.
        tokenizer (Tokenizer): Tokenizer instance for character-to-index conversion.
        context_length (int): Number of preceding characters used as context for prediction.
    """

    def __init__(self, device='cpu', tokenizer=None, context_lenght=3, words=None, generator=None):
        """
        Initializes the dataset creator with configuration parameters.

        Args:
            device (str): The device to create tensors on. Defaults to 'cpu'.
            tokenizer (Tokenizer, optional): Tokenizer instance for character-to-index conversion.
                If None, a new Tokenizer is created from the words list.
            context_lenght (int): Number of preceding characters used as context for prediction.
                Defaults to 3.
            words (list, optional): List of words to build the dataset from.
                If None, reads from 'bi-gram language model/resources/names.txt'.
            generator (torch.Generator, optional): Seeded random number generator for reproducibility.
                If None, creates a new generator with seed 2147483647.
        """
        self.device = device
        self.generator = torch.Generator(device=self.device).manual_seed(2147483647) if generator == None else generator
        self.words = open('bi-gram language model/resources/names.txt', 'r').read().splitlines() if words == None else words
        self.tokenizer = Tokenizer(self.words) if tokenizer == None else tokenizer
        self.context_length = context_lenght

    def __call__(self, words=None):
        """
        Generates input-target tensor pairs from a list of words.

        Builds sliding-window context sequences where each window of size
        `context_length` predicts the next character index. The context is
        initialized with zeros (the '.' token) and slides one position at a time.

        Args:
            words (list, optional): List of words to convert into dataset pairs.
                If None, prints an error message and returns early.

        Returns:
            list: A two-element list [X, Y] where:
                - X (torch.Tensor): Input tensor of shape (N, context_length) containing
                  character index sequences (contexts).
                - Y (torch.Tensor): Target tensor of shape (N,) containing the next
                  character index for each context.
        """
        if words == None:
            print("the words dataset is null")
            pass
        block_size = self.context_length # context length: how many characters do we take to predict the next one?
        X, Y = [], []
        try:
            for w in words:
                context = [0] * block_size
                for ch in w + '.':
                    ix = self.tokenizer.stoi[ch]
                    X.append(context)
                    Y.append(ix)
                    # print(''.join(self.tokenizer.itos[i] for i in context), '--->', self.tokenizer.itos[ix])
                    context = context[1:] + [ix] # crop and append
        
            X = torch.tensor(X, device=self.device)
            Y = torch.tensor(Y, device=self.device)
        except Exception as e:
            print(f"Failed to create dataset with exception: {e}")

        return [X, Y]