import os

from tokenizer.tokenizer import Tokenizer

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class GPT2:
    def __init__(self):
        self.hyperparamters = {
            "compression_ratio": 1.5
        }

        self.tokenizer = Tokenizer(self.hyperparamters["compression_ratio"])
        pass

    def __call__(self):
        text1 = "Ｕｎｉｃｏｄｅ! 🅤🅝🅘🅒🅞🅓🅔‽ 🇺‌🇳‌🇮‌🇨‌🇴‌🇩‌🇪! 😄 The very name strikes fear and awe into the hearts of programmers worldwide. We all know we ought to “support Unicode” in our software (whatever that means—like using wchar_t for all the strings, right?). But Unicode can be abstruse, and diving into the thousand-page Unicode Standard plus its dozens of supplementary annexes, reports, and notes can be more than a little intimidating. I don’t blame programmers for still finding the whole thing mysterious, even 30 years after Unicode’s inception."
        text2 = "Hello world"
        encoded = self.tokenizer.encode(text1)
        decoded = self.tokenizer.decode(encoded=encoded)

    def validate_model(self):
        pass

    def train_model(
        self, 
        training_src_dataset, 
        training_target_dataset, 
        validation_src_dataset, 
        validation_target_dataset
    ):
        pass

    def calculate_loss(self, logits):
        pass