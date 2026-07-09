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
    
    def encode(self, text):
        """
        Convert text to a list of indices.
        
        Args:
            text (str): Input text to encode
            
        Returns:
            list: List of integer indices representing the text
        """
        return [self.stoi[c] for c in text]
    
    def decode(self, indices):
        """
        Convert a list of indices back to text.
        
        Args:
            indices (list): List of integer indices to decode
            
        Returns:
            str: Decoded text string
        """
        return ''.join([self.itos[i] for i in indices])