import torch
import torch.nn.functional as F
from embedding_layer import *
from hidden_layer import *
from softmax_layer import *
from create_dataset import *
from batch_normalization import *
import random

# Check if MPS (Metal Performance Shaders) is available for Apple Silicon
device = torch.device("cuda" if torch.cuda.is_available()
                      else "mps" if torch.backends.mps.is_available()
                         else "cpu")

class NeuralProbabilisticLanguageModel:
    """
    A Neural Probabilistic Language Model implementation using PyTorch.

    This model predicts the next word in a sequence based on a fixed-length context 
    of previous words, utilizing an embedding layer, a hidden layer, and a softmax layer.
    """

    def __init__(self):
        """
        Initializes the NeuralProbabilisticLanguageModel with hyperparameters and layers.

        Sets up the tokenizer, dataset creation, and the three main architectural components:
        EmbeddingLayer, HiddenLayer, and SoftmaxLayer.
        """

        self.words = open('neural probabilistic language model/resources/names.txt', 'r').read().splitlines()
        self.tokenizer = Tokenizer(self.words)

        ###########################################################################
        ################# Tweak below parameters to min-max loss ##################
        ###########################################################################

        # below three variables are used to control the number of paramters
        # tweek these parameters to increase the number of parameters
        self.context_length = 5
        self.num_of_neuron_for_hidden_layer = 250
        self.feature_dimension = 12

        # parameters to control the model's training
        self.learning_iteration = 400000
        self.acceptable_loss = 1.5
        self.learning_rate = -0.1
        self.training_batch_size = 128
        self.learning_rate_decay_check_interval = 10000
        self.learning_rate_decay_check_interval_decrementer = 100
        self.number_of_words_to_sample_from_model = 20
        self.l2_lambda = 1e-4  # L2 regularization strength (weight decay coefficient)

        ###########################################################################
        ###########################################################################
        ###########################################################################

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
        
        self.batch_normalization = BatchNormalization(device=device,
                                                      num_neuron=self.num_of_neuron_for_hidden_layer)

        self.parameters = [self.embedding_layer.C, 
                           self.hidden_layer.W, 
                        #    self.hidden_layer.B, 
                           self.softmax_layer.W, 
                           self.softmax_layer.B,
                           self.batch_normalization.batch_normalization_bias,
                           self.batch_normalization.batch_normalization_gain]
        
        ########################################
        ############ model metadata ############
        ########################################
        self.training_loss = None
        self.validation_loss = None
        self.test_loss = None
        ########################################
        ########################################
        ########################################

    def __call__(self, train_model=False, validate_model=False, test_model=False):
        """
        Executes the model's primary workflow (training, validation, and testing).

        Args:
            train_model (bool): Whether to train the model on the training set.
            validate_model (bool): Whether to evaluate the model on the validation set.
            test_model (bool): Whether to evaluate the model on the test set.
        """

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
            self.validation_loss = self.test_model(X=Xdev, Y=Ydev)
            print(f"loss on validation datset is: {self.validation_loss}\n")
        if test_model:
            print("************* test start *************\n")
            self.test_loss = self.test_model(X=Xte, Y=Yte)
            print(f"loss on test dataset is: {self.test_loss}\n")

    def test_model(self, X, Y):
        """
        Evaluates the model's performance on a given dataset.

        Args:
            X (torch.Tensor): Input context indices.
            Y (torch.Tensor): Target word indices.

        Returns:
            torch.Tensor: The calculated loss for the provided data.
        """

        #Forward pass
        embedding = self.embedding_layer(X)
        embedding_concate = self.hidden_layer.concate_embedding(embedding=embedding)
        embedding_concate = self.batch_normalization(embedding_concate=embedding_concate, training=False)
        hidden_layer_output = self.hidden_layer(embedding_concate=embedding_concate)
        logits = self.softmax_layer(hidden_layer=hidden_layer_output)
        loss = self.calculate_loss(logits=logits, 
                                    Y=Y)
        return loss

    def train_model(self, X, Y):
        """
        Trains the model using stochastic gradient descent.

        Iteratively samples batches from the training data, performs forward and backward passes,
        and updates weights. Includes logic for learning rate decay and early stopping.

        Args:
            X (torch.Tensor): Training input context indices.
            Y (torch.Tensor): Training target word indices.
        """

        count = 1
        prev_loss = 100.0
        while count <= self.learning_iteration:
            ix = torch.randint(0, X.shape[0], (self.training_batch_size,))

            #Forward pass
            #-------embbed------------#
            embedding = self.embedding_layer(X[ix])
            #-------------------------#
            #------Hidden layer-------#
            embedding_concate = self.hidden_layer.concate_embedding(embedding=embedding)
                #-------batch normalization-------#
            embedding_concate = self.batch_normalization(embedding_concate=embedding_concate, training=True)
                #---------------------------------#
            hidden_layer_output = self.hidden_layer(embedding_concate=embedding_concate)
            #-------------------------#
            #-------Softmax-----------#
            logits = self.softmax_layer(hidden_layer=hidden_layer_output)
            #-------------------------#
            loss = self.calculate_loss(logits=logits, 
                                    Y=Y[ix])
            
            if count % 1000 == 0:
                print(f"current iteration is: {count} | current loss is: {loss} | current learning rate is: {-self.learning_rate}")
            
            #Check if we want learning rate decay
            if count % self.learning_rate_decay_check_interval == 0:
                self.learning_rate_decay(count=count, prev_loss=prev_loss, loss=loss.item())

            # break early if acceptable loss is reached
            if loss <= self.acceptable_loss:
                print(f"reached acceptable model loss.\ncurrent loss is: {loss}")
                self.training_loss = loss
                break
            
            #Backwards pass
            for p in self.parameters:
                if p.grad is not None:
                    p.grad.zero_()
            
            loss.backward()

            #Update weights with learning rate decay based on loss improvement
            for p in self.parameters:
                if p.grad is not None: # Safety check
                    p.data += self.learning_rate * p.grad
            
            prev_loss = loss.item()
            count += 1
            self.training_loss = loss.item()

    def calculate_loss(self, logits, Y):
        """
        Computes the cross-entropy loss between model predictions and targets,
        with optional L2 regularization (weight decay).

        Args:
            logits (torch.Tensor): The raw output from the softmax layer.
            Y (torch.Tensor): The ground truth target indices.

        Returns:
            torch.Tensor: The computed cross-entropy loss + L2 regularization term.
        """
        # Standard cross-entropy loss
        ce_loss = F.cross_entropy(logits, Y)
        
        # L2 regularization term: sum of squared weights across all parameters
        l2_norm = sum(p.pow(2).sum() for p in self.parameters)
        l2_penalty = (self.l2_lambda / 2) * l2_norm
        
        return ce_loss + l2_penalty        
    
    def sample_from_model(self):
        """
        Generates and prints random word sequences sampled from the trained model.

        Starts with an empty context and iteratively predicts the next word until 
        an end-of-sequence token (0) is reached or a maximum length is hit.
        """
        for _ in range(self.number_of_words_to_sample_from_model):
            out = []
            context = [0] * self.context_length # initialize with all ...
            while True:
                emb = self.embedding_layer(torch.tensor([context])) # (1,block_size,d)
                emb_concat = self.hidden_layer.concate_embedding(embedding=emb)
                emb_concat_normalized = self.batch_normalization(embedding_concate=emb_concat, training=False)
                h = self.hidden_layer(embedding_concate=emb_concat_normalized)
                logits = self.softmax_layer(h)
                probs = F.softmax(logits, dim=1)
                ix = torch.multinomial(probs, num_samples=1, generator=self.generator).item()
                context = context[1:] + [ix]
                out.append(ix)
                if ix == 0:
                    break
            
            print(''.join(self.tokenizer.itos[i] for i in out))

    def learning_rate_decay(self, count, loss, prev_loss):
        """
        Adjusts the learning rate based on the improvement of the loss.

        If the loss improvement is below a certain threshold, the learning rate is decreased.
        Conversely, if the loss increases or improves very little, it may be adjusted 
        to escape local minima or stabilize training.

        Args:
            count (int): Current iteration number.
            loss (float): Current batch loss.
            prev_loss (float): Loss from the previous check interval.
        """
        # Apply learning rate decay if loss is not improving significantly
        if count > 5000:  # Only start decay after some iterations
            loss_improvement = (prev_loss - loss) / prev_loss if prev_loss != 0 else 0
            if loss_improvement < 0.01:  # If improvement is less than 0.1%
                self.learning_rate *= 0.95  # Reduce learning rate
                if self.learning_rate_decay_check_interval > 5000:
                    self.learning_rate_decay_check_interval -= self.learning_rate_decay_check_interval_decrementer
            if loss_improvement > 0.01: # If loss increases instead of decreasing
                self.learning_rate /= 0.999 # Increase learning rate
                if self.learning_rate_decay_check_interval > 5000:
                    self.learning_rate_decay_check_interval -= self.learning_rate_decay_check_interval_decrementer