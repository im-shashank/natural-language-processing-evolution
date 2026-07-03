import torch
import torch.nn.functional as F
from embedding_layer import *
from hidden_layer import *
from softmax_layer import *
from create_dataset import *
import random

# Check if MPS (Metal Performance Shaders) is available for Apple Silicon
device = torch.device("cuda" if torch.cuda.is_available()
                      else "mps" if torch.backends.mps.is_available()
                         else "cpu")

class NeuralProbabilisticLanguageModel:
    def __init__(self):
        self.words = open('bi-gram language model/resources/names.txt', 'r').read().splitlines()
        self.tokenizer = Tokenizer(self.words)

        # below three variables are used to control the number of paramters
        # tweek these parameters to increase the number of parameters
        self.context_length = 5
        self.num_of_neuron_for_hidden_layer = 300
        self.feature_dimension = 12

        # parameters to control the model's training
        self.learning_iteration = 400000
        self.acceptable_loss = 1.8
        self.learning_rate = -0.1
        self.training_batch_size = 5000
        self.learning_rate_decay_check_interval = 10000
        self.learning_rate_decay_check_interval_decrementer = 100

        # a manual generator to keep the outputs same throughtout multiple runs
        self.generator = torch.Generator(device=device).manual_seed(2147483647)

        self.create_data_set = CreateDataset(device=device,
                                             tokenizer=self.tokenizer,
                                             context_lenght=self.context_length,
                                             words=self.words,
                                             generator=self.generator)

        self.embedding_layer = EmbeddingLayer(device=device, 
                                              tokenizer=self.tokenizer, 
                                              feature_dimension=self.feature_dimension, 
                                              words=self.words,
                                              generator=self.generator)
        
        self.hidden_layer = HiddenLayer(device=device, 
                                        context_lenght=self.context_length, 
                                        feature_dimension=self.feature_dimension,
                                        num_neuron=self.num_of_neuron_for_hidden_layer,
                                        generator=self.generator)
        
        self.softmax_layer = SoftmaxLayer(device=device, 
                                          tokenizer=self.tokenizer,
                                          num_neuron=self.num_of_neuron_for_hidden_layer,
                                          generator=self.generator)

        self.parameters = [self.embedding_layer.C, 
                           self.hidden_layer.W, 
                           self.hidden_layer.B, 
                           self.softmax_layer.W, 
                           self.softmax_layer.B]

    def __call__(self, train_model=False, validate_model=False, test_model=False):
        random.seed(42)
        random.shuffle(self.words)
        n1 = int(0.8*len(self.words))
        n2 = int(0.9*len(self.words))

        Xtr, Ytr = self.create_data_set(self.words[:n1])
        Xdev, Ydev = self.create_data_set(self.words[n1:n2])
        Xte, Yte = self.create_data_set(self.words[n2:])

        if train_model:
            print("************* training start *************\n")
            self.train_model(X=Xtr, Y=Ytr)
            print("************* training end ***************\n")
        if validate_model:
            print("************* validating model *************\n")
            loss = self.test_model(X=Xdev, Y=Ydev)
            print(f"loss on validation datset is: {loss}\n")
        if test_model:
            print("************* test start *************\n")
            loss = self.test_model(X=Xte, Y=Yte)
            print(f"loss on test dataset is: {loss}\n")
        print(f"total number of parameters is: {sum(p.nelement() for p in self.parameters)}")

    def test_model(self, X, Y):
        #Forward pass
        embedding = self.embedding_layer(X)
        hidden_layer_output = self.hidden_layer(embedding=embedding)
        logits = self.softmax_layer(hidden_layer=hidden_layer_output)
        loss = self.calculate_loss(logits=logits, 
                                    Y=Y)
        return loss

    def train_model(self, X, Y):
        count = 1
        prev_loss = 100.0
        while count <= self.learning_iteration:
            ix = torch.randint(0, X.shape[0], (self.training_batch_size,))

            #Forward pass
            embedding = self.embedding_layer(X[ix])
            hidden_layer_output = self.hidden_layer(embedding=embedding)
            logits = self.softmax_layer(hidden_layer=hidden_layer_output)
            loss = self.calculate_loss(logits=logits, 
                                    Y=Y[ix])
            
            if count % 1000 == 0:
                print(f"current iteration is: {count} | current loss is: {loss} | current learning rate is: {-self.learning_rate}")
            
            #Check if we want learning rate decay
            if count % self.learning_rate_decay_check_interval == 0:
                self.learning_rate_decay(count=count, prev_loss=prev_loss, loss=loss)

            # break early if acceptable loss is reached
            if loss <= self.acceptable_loss:
                print(f"reached acceptable model loss.\ncurrent loss is: {loss}")
                break
            
            #Backwards pass
            for p in self.parameters:
                p.grad = None
            
            loss.backward()

            #Update weights with learning rate decay based on loss improvement
            for p in self.parameters:
                p.data += self.learning_rate * p.grad
            
            prev_loss = loss
            count += 1

    def calculate_loss(self, logits, Y):
        loss = F.cross_entropy(logits, Y)
        return loss        
    
    def sample_from_model(self):
        #TODO
        pass

    def learning_rate_decay(self, count, loss, prev_loss):
        # Apply learning rate decay if loss is not improving significantly
        if count > 1000:  # Only start decay after some iterations
            loss_improvement = (prev_loss - loss) / prev_loss if prev_loss != 0 else 0
            if loss_improvement < 0.001:  # If improvement is less than 0.1%
                self.learning_rate *= 0.999  # Reduce learning rate by 5%
                if self.learning_rate_decay_check_interval > 5000:
                    self.learning_rate_decay_check_interval -= self.learning_rate_decay_check_interval_decrementer
            if loss_improvement > 0.001: # If loss increases instead of decreasing
                self.learning_rate /= 0.999 # Increase learning rate by 5%
                if self.learning_rate_decay_check_interval > 5000:
                    self.learning_rate_decay_check_interval -= self.learning_rate_decay_check_interval_decrementer