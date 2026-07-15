class Tokenizer:

    def __init__(self, text):
        self.chars = sorted(list(set(''.join(text))))
        self.stoi = { ch:i for i,ch in enumerate(self.chars) }
        self.itos = {i: s for s, i in self.stoi.items()}

    def __call__(self):
        print(self.chars)
        print(len(self.chars))

    def encode(self, text):
        encoded = [self.stoi[c] for c in text] # encoder: take a string, output a list of integers
        return encoded
    
    def decode(self, indices):
        decoded = ''.join([self.itos[i] for i in indices]) # decoder: take a list of integers, output a string
        return decoded