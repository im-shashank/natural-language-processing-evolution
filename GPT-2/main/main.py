import os

from model.gpt_2 import GPT2

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model = GPT2()
model()
