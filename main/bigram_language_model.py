import torch
import torch.nn.functional as F
from bigram_neuron import *
from training_set import *

# Check if MPS (Metal Performance Shaders) is available for Apple Silicon
device = torch.device("cuda" if torch.cuda.is_available()
                      else "mps" if torch.backends.mps.is_available()
                         else "cpu")
print(f"Training on device: {device}")

class BigramLanguageModel:
    """
    A class to represent a bigram language model.

    This model predicts the next character based on the previous character
    using character-level bigram probabilities.
    
    Attributes:
        trainingSet (TrainingSet): Instance of TrainingSet for data preparation.
        N (torch.Tensor): Bigram count matrix.
        P (torch.Tensor): Probability matrix derived from counts.
        learning_rate [int32]: learning rate of bigram model.
    """

    def __init__(self):
        """
        Initializes the BigramLanguageModel instance.
        
        Sets up the training set and initializes the bigram count matrix.
        """
        self.trainingSet = TrainingSet(device)
        self.N = torch.zeros((27, 27), dtype=torch.int32)
        self.P = None
        self.learning_rate = 0.1
        self.neuron = BiGramNeuron(device)
        self.acceptable_loss = 1

    def train_model(self):
        """
        Trains the model using the neural network approach.
        
        This method creates a training dataset, initializes a neuron,
        computes probabilities, and calculates the loss.
        
        Returns:
            None
        """
        
        training_dataset = self.trainingSet.create_training_set_for_neural_net()
        xenc = training_dataset[0]
        ys = training_dataset[1]
        num_elements = training_dataset[2]

        i = 1
        prev_loss = None
        while True and i < 100000:
            probs = self.neuron(xenc)
            loss = self.calculate_loss(probs, ys, self.neuron, num_elements)
            if i%100 == 0:
                print(f"current loss is : {loss.item()} | current leaning rate is: {self.learning_rate} | current iterator count is: {i}")
                
                # EARLY STOPPING: If loss hasn't changed by at least 0.0001, stop training
                if prev_loss is not None and abs(prev_loss - loss.item()) < 0.00001:
                    print("Training converged early. Stopping.")
                    break
                prev_loss = loss.item()

            if loss.item() < self.acceptable_loss:
                break

            # Clear gradients BEFORE computing new ones
            self.neuron.W.grad = None  # Proper way to clear gradients

            # calculate the loss
            loss.backward()

            # update the weights
            self.neuron.W.data += -self.learning_rate * self.neuron.W.grad

            i += 1
        
    
    def calculate_loss(self, probs, ys, neuron, num_elements):
        """
        Calculates the average negative log-likelihood loss.
        
        Args:
            probs (torch.Tensor): Probability tensor from the model.
            ys (torch.Tensor): True labels (next character indices).
            
        Returns:
            torch.Tensor: The calculated loss value.
        """
        loss = -probs[torch.arange(num_elements), ys].log().mean() + 0.01*(neuron.W**2).mean()
        return loss

    def generate_words(self, num_words=5):
        """
        Generates a list of words based on the trained neural network's probabilities.
        """
        # Ensure the random generator is on the correct device
        g = torch.Generator(device=device).manual_seed(2147483647)
        words = []

        for _ in range(num_words):
            out = []
            ix = 0  # Starting character index (0 represents '.')
            
            while True:
                # 1. Create a one-hot encoded tensor for the current character
                xenc = F.one_hot(torch.tensor([ix], device=device), num_classes=27).float()
                
                # 2. Pass it through the trained neuron to get the probability distribution
                probs = self.neuron(xenc)
                
                # 3. Sample the next character based on the probabilities
                ix = torch.multinomial(probs, num_samples=1, replacement=True, generator=g).item()
                
                # 4. Stop if we sample the end character '.' (index 0)
                if ix == 0:
                    break
                
                # 5. Map the index back to a string character and append it
                out.append(self.trainingSet.tokenizer.itos[ix])
            
            words.append(''.join(out))

        return words
