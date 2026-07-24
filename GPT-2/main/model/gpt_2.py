import os
import sys
import torch
import torch.nn.functional as F
import torch.nn as nn
import datetime
import csv

from torch.utils.data import DataLoader
from datasets import load_dataset
from tqdm import tqdm

current_dir = os.path.dirname(os.path.abspath(__file__))
main_dir = os.path.abspath(os.path.join(current_dir, ".."))
if main_dir not in sys.path:
    sys.path.insert(0, main_dir)

from tokenizer.tokenizer import Tokenizer
from model.transformer.transformer import Transformer
from plot.plotting import Plotting
from model.weight_manager import WeightManager

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Check if MPS (Metal Performance Shaders) is available for Apple Silicon
device = torch.device("cuda" if torch.cuda.is_available()
                      else "mps" if torch.backends.mps.is_available()
                         else "cpu")

class GPT2:
    """A GPT-2 language model implementation.
    
    This class orchestrates the tokenizer, model training, validation, and loss calculation.
    """
    
    def __init__(self):
        """Initializes the GPT-2 model hyperparameters and tokenizer."""
        self.hyperparamters = {
            ########### tokenizer hyperparameter ############
            "compression_ratio": 2.0,
            "training_cutoff_metric": "compression_ratio",
            "save_tokenizer_weights": False,
            #################################################

            ########### tranformer hyperparamters ###########
            "vocab_size": 50304,
            "d_model": 768,
            "max_len": 1024,
            "n_layers": 12,
            "n_heads": 12,
            "d_ff": 3072,
            "device": device,
            "dropout": 0.1,
            #################################################

            ########### optimizer & scheduler hyperparamters ############
            "learning_rate": 3e-4,
            "weight_decay": 0.1,
            "num_steps": 100000,
            "div_factor": 25.0,
            "final_div_factor": 1e4,
            "pct_start": 0.1,
            #############################################################

            ########### dataloader hyperparamters ###########
            "batch_size": 16,
            "validation_dataset_size": 100000, # 100000 seems to be a good number
            #################################################
        }

        self.tokenizer = Tokenizer(
            desired_vocab_size=self.hyperparamters["vocab_size"],
            save_tokenizer_weights=self.hyperparamters["save_tokenizer_weights"]
        )

        self.model = Transformer(
            vocab_size=self.hyperparamters["vocab_size"],
            d_model=self.hyperparamters["d_model"],
            max_len=self.hyperparamters["max_len"],
            n_layers=self.hyperparamters["n_layers"],
            n_heads=self.hyperparamters["n_heads"],
            d_ff=self.hyperparamters["d_ff"],
            device=self.hyperparamters["device"],
            dropout=self.hyperparamters["dropout"]
        )
        
        self.plotting = Plotting()
        self.weight_manager = WeightManager()

        self.training_loss = 0.0
        self.validation_loss = 0.0

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.hyperparamters["learning_rate"],
            weight_decay=self.hyperparamters["weight_decay"]
        )

        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer=self.optimizer,
            max_lr=self.hyperparamters["learning_rate"],
            total_steps=self.hyperparamters["num_steps"],
            div_factor=self.hyperparamters["div_factor"],
            final_div_factor=self.hyperparamters["final_div_factor"],
            pct_start=self.hyperparamters["pct_start"]
        )

    def __call__(self, train=False, validate=False):
        train_dataloader, validation_dataloader = self.data_loader()

        print(f"total number of parameters: {sum(parameter.nelement() for parameter in self.model.parameters())}")

        if train:
            self.train_model(train_dataloader)
            self.plotting.plot()
        
        if validate:
            self.validate_model(validation_dataloader)

    @torch.no_grad()
    def validate_model(self, validation_dataloader):
        """Validates the model against a validation dataset."""
        self.model.eval()

        total_loss = 0.0
        num_batches = 0

        print("Starting validation...")
        for batch in tqdm(validation_dataloader, desc="Validating GPT-2"):
            text_batch = batch["text"]
            x, y = self.process_batch(text_batch, self.tokenizer, self.hyperparamters["max_len"])
            
            logits = self.model(x)
            loss = self.calculate_loss(logits, y)
            
            total_loss += loss.item()
            num_batches += 1

        average_loss = total_loss / num_batches if num_batches > 0 else 0.0
        
        self.plotting.track_val(average_loss)
        self.model.train()
        
        self.validation_loss = average_loss
        print(f"Validation complete. Average Loss: {average_loss:.4f}")
        return average_loss

    def train_model(self, train_dataloader):
        # Setup logging
        log_dir = os.path.join(base_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file_path = os.path.join(log_dir, f"train_log_{timestamp}.log")
        with open(log_file_path, "w") as f:
            f.write("Step,Training Loss,Learning Rate\n")

        optimizer = self.optimizer
        scheduler = self.scheduler

        train_iter = iter(train_dataloader)

        progress_bar = tqdm(total=self.hyperparamters["num_steps"], desc="Training GPT-2")

        for step in range(self.hyperparamters["num_steps"]):
            try:
                batch = next(train_iter)
            except StopIteration:
                # If we miraculously reach the end of the massive dataset, 
                # just restart the iterator for the next epoch!
                train_iter = iter(train_dataloader)
                batch = next(train_iter)
            
            text_batch = batch["text"]

            # prepare x and y for training
            x, y = self.process_batch(text_batch, self.tokenizer, self.hyperparamters["max_len"])

            # zero the gradients
            optimizer.zero_grad()

            # forward pass
            logits = self.model(x)

            # calculate loss and track it for making graph
            loss = self.calculate_loss(logits, y)
            self.plotting.track_train(loss)
            self.training_loss = loss.item()

            # update progress bar
            progress_bar.update(1)
            current_lr = scheduler.get_last_lr()[0]
            progress_bar.set_postfix({"Training Loss": f"{loss.item():.4f}", "Learning Rate": f"{current_lr:.4e}"})

            #log the training
            if step % 100 == 0:
                with open(log_file_path, "a") as f:
                    f.write(f"{step},{loss.item():.4f},{current_lr:.4e}\n")
            
            loss.backward()

            optimizer.step()
            scheduler.step()

            if step > 0 and step % 10000 == 0:
                self.weight_manager.save(
                    model=self.model,
                    optimizer=optimizer,
                    step=step
                )

        progress_bar.close()

    def calculate_loss(self, logits, target_set):
        return F.cross_entropy(logits.reshape(-1, logits.size(-1)), target_set.reshape(-1))

    def process_batch(self, text_batch, tokenizer, max_len=1024):
        # text_batch is a list of strings (batch_size=16)
        encoded_list = tokenizer.encode_batch(text_batch)
        
        # We need all sequences to be exactly max_len + 1 tokens long.
        # <|endoftext|> is index 0 in our BPE tokenizer.
        pad_id = 0 
        
        batch_tokens = []
        for encoded in encoded_list:
            if len(encoded) >= max_len + 1:
                # Truncate if it's too long
                batch_tokens.append(encoded[:max_len + 1])
            else:
                # Pad if it's too short
                batch_tokens.append(encoded + [pad_id] * ((max_len + 1) - len(encoded)))
                
        # Now batch_tokens is a perfectly uniform 2D list of shape (batch_size, max_len + 1)
        tokens = torch.tensor(batch_tokens, dtype=torch.long, device=self.hyperparamters["device"])
        
        # x gets the first max_len tokens, y gets the last max_len tokens (shifted by 1)
        x = tokens[:, :-1]  
        y = tokens[:, 1:]   
        return x, y

    def data_loader(self, validation_dataset_size=100000):
        dataset = load_dataset("Skylion007/openwebtext", split="train", streaming=True)

        validation_dataset = dataset.take(validation_dataset_size)
        train_dataset = dataset.skip(validation_dataset_size)

        train_dataset = train_dataset.shuffle(seed=42, buffer_size=10000)

        # Convert dataset to batches
        train_loader = DataLoader(train_dataset, batch_size=self.hyperparamters["batch_size"])
        validation_loader = DataLoader(validation_dataset, batch_size=self.hyperparamters["batch_size"])

        return train_loader, validation_loader

    def export_run_to_csv(self, optimizer_name="AdamW", filename="experiment_logs.csv"):
        filepath = os.path.join(base_dir, filename)
        file_exists = os.path.isfile(filepath)

        run_data = {
            "Timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Total_Parameters": sum(p.nelement() for p in self.model.parameters()),
            "Vocab_Size": self.hyperparamters["vocab_size"],
            "D_Model": self.hyperparamters["d_model"],
            "Max_Len": self.hyperparamters["max_len"],
            "N_Layers": self.hyperparamters["n_layers"],
            "N_Heads": self.hyperparamters["n_heads"],
            "D_FF": self.hyperparamters["d_ff"],
            "Dropout": self.hyperparamters["dropout"],
            "Batch_Size": self.hyperparamters["batch_size"],
            "Max_Iterations": self.hyperparamters["num_steps"],
            "Optimizer": optimizer_name,
            "Learning_Rate": self.hyperparamters["learning_rate"],
            "Weight_Decay": self.hyperparamters["weight_decay"],
            "Train_Loss": round(self.training_loss, 4),
            "Validation_Loss": round(self.validation_loss, 4),
        }

        with open(filepath, mode='a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=run_data.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(run_data)

        print(f"\n---> Run data successfully exported to {filepath} <---")

    @torch.no_grad()
    def sample(self, prompt: str, max_new_tokens: int = 200, temperature: float = 1.0, top_k: int | None = None, output_to_file: bool = False, token_to_stop = [True, '<|endoftext|>']):
        self.model.eval()

        generated = list(self.tokenizer.encode(prompt))
        
        # Determine the stop token ID
        stop_token_id = None
        if token_to_stop[0] and isinstance(token_to_stop[1], str):
            encoded_stop = self.tokenizer.encode(token_to_stop[1])
            if encoded_stop:
                stop_token_id = encoded_stop[0]
            elif token_to_stop[1] == '<|endoftext|>':
                stop_token_id = 0

        for _ in range(max_new_tokens):
            context = generated[-self.hyperparamters["max_len"]:]
            context_t = torch.tensor(context, dtype=torch.long, device=self.hyperparamters["device"]).unsqueeze(0)

            logits = self.model(context_t)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                top_vals, _ = torch.topk(logits, top_k)
                logits[logits < top_vals[:, -1:]] = float('-inf')

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()
            
            if stop_token_id is not None and next_token == stop_token_id:
                break
            generated.append(next_token)

        self.model.train()

        if output_to_file:
            resources_dir = os.path.join(base_dir, "resources")
            os.makedirs(resources_dir, exist_ok=True)
            filename = f"output_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.txt"
            with open(os.path.join(resources_dir, filename), "w") as file:
                file.write(self.tokenizer.decode(generated))
            return f"{filename} successfully generated"
        else:
            return self.tokenizer.decode(generated)