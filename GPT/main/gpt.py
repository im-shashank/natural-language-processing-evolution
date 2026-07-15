from tokenizer import *
from create_dataset import *

torch.manual_seed(1337)

# Check if MPS (Metal Performance Shaders) is available for Apple Silicon
device = torch.device("cuda" if torch.cuda.is_available()
                      else "mps" if torch.backends.mps.is_available()
                         else "cpu")

class GPT:

    def __init__(self):
        ###########################################################################
        ################# Tweak below parameters to min-max loss ##################
        ###########################################################################

        self.batch_size = 4 # how many independent sequences will we process in parallel?
        self.block_size = 8 # what is the maximum context length for predictions?

        ###########################################################################
        ###########################################################################
        ###########################################################################

        self.text = open("resources/input.txt","r").read()

        self.tokenizer = Tokenizer(self.text)

        self.create_dataset = CreateDataset(tokenizer=self.tokenizer, device=device)

    def __call__(self, train=True, validate=True):
        dataset = self.create_dataset(self.text)

        n = int(0.9*len(dataset))
        train_data = dataset[:n]
        val_data = dataset[n:]

        if train:
            Xb, Yb = self.get_batch(split="train", train_data=train_data, val_data=val_data)

    def get_batch(self, split, train_data, val_data):
        # generate a small batch of data of inputs x and targets y
        data = train_data if split == 'train' else val_data
        ix = torch.randint(len(data) - self.block_size, (self.batch_size,))
        x = torch.stack([data[i:i+self.block_size] for i in ix])
        y = torch.stack([data[i+1:i+self.block_size+1] for i in ix])
        return x, y