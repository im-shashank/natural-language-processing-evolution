from sympy import true
import os

from model.gpt_2 import GPT2

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

gpt2 = GPT2()

# Load weights before training
weight_loaded = gpt2.weight_manager.load(
    model=gpt2.model,
    optimizer=gpt2.optimizer,
    device=gpt2.hyperparamters["device"]
)

if weight_loaded == None:
    # Run training
    gpt2(train=True, validate=True)

    # Save final weights after training finishes
    gpt2.weight_manager.save(
        model=gpt2.model,
        optimizer=gpt2.optimizer,
        step=gpt2.hyperparamters["num_steps"]
    )
    
    # Export run to csv
    gpt2.export_run_to_csv()
else:
    gpt2(train=False, validate=True)

print(gpt2.sample(prompt="Hello, my name is"))