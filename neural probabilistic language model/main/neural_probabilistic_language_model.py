import torch
import torch.nn.functional as F
from embedding_layer import *
from hidden_layer import *
from softmax_layer import *

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
        self.context_length = 3
        self.num_of_neuron_for_hidden_layer = 200
        self.feature_dimension = 10

        self.embedding_layer = EmbeddingLayer(device=device, 
                                              context_lenght=self.context_length, 
                                              tokenizer=self.tokenizer, 
                                              feature_dimension=self.feature_dimension, 
                                              words=self.words)
        self.hidden_layer = HiddenLayer(device=device, 
                                        context_lenght=self.context_length, 
                                        feature_dimension=self.feature_dimension,
                                        num_neuron=self.num_of_neuron_for_hidden_layer)
        self.softmax_layer = SoftmaxLayer(device=device, 
                                          tokenizer=self.tokenizer,
                                          num_neuron=self.num_of_neuron_for_hidden_layer)

        self.parameters = [self.embedding_layer.C, 
                           self.hidden_layer.W, 
                           self.hidden_layer.B, 
                           self.softmax_layer.W, 
                           self.softmax_layer.B]

        self.learning_rate = 1
    
    def train_model(self):
        embedding = self.embedding_layer()
        hidden_layer_output = self.hidden_layer(embedding=embedding[2])
        probabilities = self.softmax_layer(hidden_layer=hidden_layer_output)
        loss = self.calculate_loss(probabilities=probabilities, 
                                   probabilities_size=probabilities.shape[0], 
                                   Y=embedding[1])
        print(f"current loss = {loss}")
        print(f"total number of parameters = {sum(p.nelement() for p in self.parameters)}")

    def calculate_loss(self, probabilities, probabilities_size, Y):
        loss = -probabilities[torch.arange(probabilities_size), Y].log().mean()
        return loss
    
    def sample_from_model(self):
        pass
