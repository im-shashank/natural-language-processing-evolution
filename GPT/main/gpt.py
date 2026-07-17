import torch
from torch.nn import parameter
import torch.nn.functional as F
import os
import csv
from datetime import datetime

from tokenizer import *
from create_dataset import *
from transformer.transformer import Transformer
from plotting import Plotting
from weight_manager import WeightManager

torch.manual_seed(1337)

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Check if MPS (Metal Performance Shaders) is available for Apple Silicon
device = torch.device("cuda" if torch.cuda.is_available()
                      else "mps" if torch.backends.mps.is_available()
                         else "cpu")

class GPT:

    def __init__(self):
        ###########################################################################
        ################# Tweak below parameters to min-max loss ##################
        ###########################################################################

        self.batch_size = 16 # how many independent sequences will we process in parallel?
        self.d_model = 128 # what is the maximum context length for predictions?
        self.max_len = 128 # maximum length for positional encoding
        self.n_layers = 6 # number of layers in the transformer
        self.n_heads = 8 # number of heads in the transformer
        self.d_ff = 2048 # dimension of the feedforward network
        self.dropout = 0.1 # dropout rate

        self.training_batch_size = 64
        self.learning_rate = 3e-4
        self.weight_decay = 0.1

        self.learning_iteration = 10000
        self.eval_iterations = 20 # number of batches to average when estimating validation loss0

        ###########################################################################
        ###########################################################################
        ###########################################################################

        self.text = open(os.path.join(base_dir, "resources", "input.txt"),"r").read()

        self.tokenizer = Tokenizer(self.text)

        self.vocab_size = len(self.tokenizer.stoi)

        self.create_dataset = CreateDataset(tokenizer=self.tokenizer, device=device)

        self.model = Transformer(
                src_vocab_size=self.vocab_size,
                tgt_vocab_size=self.vocab_size,
                d_model=self.d_model,
                max_len=self.max_len,
                n_layers=self.n_layers,
                n_heads=self.n_heads,
                d_ff=self.d_ff,
                dropout=self.dropout,
                device=device
            )
        
        self.plotting = Plotting()
        self.weight_manager = WeightManager()

        self.validation_loss = 0
        self.training_loss = 0

    def __call__(self, train=True, validate=True):
        dataset = self.create_dataset(self.text)

        n = int(0.9*len(dataset))
        self.train_data = dataset[:n]
        self.val_data = dataset[n:]

        print(f"total number of parameters: {sum( p.nelement() for p in self.model.parameters())}")

        if train:
            self.train_model()
            self.plotting.plot()
        if validate:
            self.validate_model()

        if train:
            print(f"training loss: {self.training_loss}, validation loss: {self.validation_loss}")
        else:
            print(f"validation loss: {self.validation_loss}")
    
    @torch.no_grad()
    def validate_model(self):
        self.model.eval() # tell the layers like BatchNorm1D to lock their state

        # Average over several fresh validation batches for a stable estimate.
        losses = []
        for _ in range(self.eval_iterations):
            X_val_b, Y_val_b = self.get_batch(self.val_data)
            # Feed the context X to both encoder and decoder; Y (X shifted by 1) is only the target.
            logits = self.model(X_val_b, X_val_b)
            losses.append(self.calculate_loss(logits=logits, target_set=Y_val_b).item())
        self.validation_loss = sum(losses) / len(losses)

        self.model.train() # Switch the model back to training mode
    
    def train_model(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=self.learning_rate,
            weight_decay=self.weight_decay
            )
        
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer=optimizer,
            max_lr=self.learning_rate,
            total_steps=self.learning_iteration,
            div_factor=25.0,
            final_div_factor=1e4,
            pct_start=0.1
        )

        for count in range(self.learning_iteration):
            # 1. sample a fresh mini batch from the FULL training data each step
            X_train_b, Y_train_b = self.get_batch(self.train_data)

            # 2. Zero the gradients from the previous step
            optimizer.zero_grad()

            # 3. forward pass (decoder sees context X, not the target Y, to avoid leaking the answer)
            logits = self.model(X_train_b, X_train_b)

            # 4. calculate loss
            loss = self.calculate_loss(logits=logits, target_set=Y_train_b)
            self.training_loss = loss.item()
            self.plotting.track_train(loss=loss) # for tracking the loss and later drawing a plot

            # 5. print the loss
            if count % 100 == 0 or count == 0 or count == self.learning_iteration - 1:
                # self.validate_model()
                # self.plotting.track_val(self.validation_loss)
                print(f"iteration: {count}/{self.learning_iteration}, training loss: {self.training_loss}")

            # 5. backpropagation throught the layers
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            # 6. Update the weights using the calculated gradients
            optimizer.step()

            # 7. Advance the OneCycleLR schedule (steps once per batch, right after optimizer.step)
            scheduler.step()

    def save_weights(self, filename="transformer.pt", optimizer=None):
        self.weight_manager.save(self.model, filename, optimizer=optimizer)

    def load_weights(self, filename="transformer.pt", optimizer=None):
        return self.weight_manager.load(self.model, filename, optimizer=optimizer, device=device)

    def export_run_to_csv(self, optimizer_name="AdamW", filename="experiment_logs.csv"):
        filepath = os.path.join(base_dir, filename)
        file_exists = os.path.isfile(filepath)

        run_data = {
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Total_Parameters": sum(p.nelement() for p in self.model.parameters()),
            "Vocab_Size": self.vocab_size,
            "D_Model": self.d_model,
            "Max_Len": self.max_len,
            "N_Layers": self.n_layers,
            "N_Heads": self.n_heads,
            "D_FF": self.d_ff,
            "Dropout": self.dropout,
            "Batch_Size": self.batch_size,
            "Training_Batch_Size": self.training_batch_size,
            "Max_Iterations": self.learning_iteration,
            "Optimizer": optimizer_name,
            "Learning_Rate": self.learning_rate,
            "Weight_Decay": self.weight_decay,
            "Train_Loss": round(self.training_loss, 4),
            "Validation_Loss": round(self.validation_loss, 4),
        }

        with open(filepath, mode='a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=run_data.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(run_data)

        print(f"\n---> Run data successfully exported to {filepath} <---")

    def calculate_loss(self, logits, target_set):
        return F.cross_entropy(logits.view(-1, logits.size(-1)), target_set.view(-1))

    def get_batch(self, data):
        # generate a batch of inputs x and targets y sampled from the whole data split.
        # window length is the context length (max_len), y is x shifted by one position.
        ix = torch.randint(len(data) - self.max_len, (self.training_batch_size,), device=device)
        x = torch.stack([data[i:i+self.max_len] for i in ix])
        y = torch.stack([data[i+1:i+self.max_len+1] for i in ix])
        return x, y
    
    @torch.no_grad()
    def sample(self, prompt: str, max_new_tokens: int = 200, temperature: float = 1.0, top_k: int | None = None, output_to_file: bool =False, token_to_stop = [True, '.']):
        self.model.eval()

        generated = list(self.tokenizer.encode(prompt))

        for _ in range(max_new_tokens):
            # Mirror training: feed the last max_len tokens as context to both encoder
            # and decoder, then read the next-token distribution from the last position.
            context = generated[-self.max_len:]
            context_t = torch.tensor(context, dtype=torch.long, device=device).unsqueeze(0)

            logits = self.model(context_t, context_t)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                top_vals, _ = torch.topk(logits, top_k)
                logits[logits < top_vals[:, -1:]] = float('-inf')

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()
            if token_to_stop[0] and next_token == self.tokenizer.stoi[token_to_stop[1]]:
                break
            generated.append(next_token)

        self.model.train()

        if output_to_file:
            with open(os.path.join(base_dir, "resources", f"output_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.txt"), "w") as file:
                file.write(self.tokenizer.decode(generated))
            return f"output_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.txt successfully generated"
        else:
            return self.tokenizer.decode(generated)