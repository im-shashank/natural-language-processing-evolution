class Tokenizer:
    """
    A class for tokenizing text data into character-level indices.
    
    This tokenizer handles the conversion between characters and numerical indices,
    including a special character for word boundaries.
    
    Attributes:
        chars (list): Sorted list of unique characters in the vocabulary.
        stoi (dict): Mapping from characters to indices.
        itos (dict): Mapping from indices to characters.
        special_char (str): Special character used for word boundaries (default: '.').
    """

    def __init__(self, words):
        """
        Initializes the Tokenizer with a list of words.
        
        Args:
            words (list): List of words to build the vocabulary from.
        """
        self.special_char = '.'
        self.chars = sorted(list(set(''.join(words))))
        self.stoi = {self.special_char: 0}
        for i, s in enumerate(self.chars):
            self.stoi[s] = i + 1
        self.itos = {i: s for s, i in self.stoi.items()}