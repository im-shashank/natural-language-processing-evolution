from tokenizers import Tokenizer as HFTokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from .tokenizer_weight_manager import TokenizerWeightsManager

class Tokenizer:
    def __init__(self, desired_vocab_size=50304, save_tokenizer_weights=False):
        self.desired_vocab_size = desired_vocab_size
        
        # Initialize the underlying Hugging Face tokenizer
        self._hf_tokenizer = HFTokenizer(BPE(unk_token=None))
        self._hf_tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
        self._hf_tokenizer.decoder = ByteLevelDecoder()

        # Let the weight manager handle loading if weights exist
        self.weight_manager = TokenizerWeightsManager(self, save_tokenizer_weights=save_tokenizer_weights)
        self.weight_manager.load_weights()

    def __call__(self, text):
        return self.encode(text)

    def train_from_iterator(self, iterator):
        """Trains the tokenizer using a Python generator (iterator)."""
        from tokenizers.trainers import BpeTrainer
        trainer = BpeTrainer(vocab_size=self.desired_vocab_size, special_tokens=["<|endoftext|>"])
        
        self._hf_tokenizer.train_from_iterator(iterator, trainer=trainer)
        self.weight_manager.save_weights()

    def encode(self, text):
        # .encode() returns an Encoding object. .ids gives the list of integer tokens.
        return self._hf_tokenizer.encode(text).ids

    def encode_batch(self, text_batch):
        # .encode_batch() handles a list of strings efficiently
        encodings = self._hf_tokenizer.encode_batch(text_batch)
        return [enc.ids for enc in encodings]

    def decode(self, encoded):
        return self._hf_tokenizer.decode(encoded)
