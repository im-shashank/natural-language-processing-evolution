from gpt import *

gpt = GPT()

if gpt.weight_manager.exists():
    gpt.load_weights()
    gpt(train=False, validate=True)
else:
    gpt()
    gpt.save_weights()
    gpt.export_run_to_csv()

print(
    f"\n{gpt.sample(prompt=" ", 
    max_new_tokens=10000, 
    temperature=0.8, 
    output_to_file=True, 
    token_to_stop=[False, '.'])}"
)