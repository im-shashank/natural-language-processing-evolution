import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizer import *
from create_dataset import *
from embedding import *
from flatten_consecutive import *
from batch_norm_1d import *
from plotting import *
from residual_block import *
import csv
import os
from datetime import datetime

# Check if MPS (Metal Performance Shaders) is available for Apple Silicon
device = torch.device("cuda" if torch.cuda.is_available()
                      else "mps" if torch.backends.mps.is_available()
                         else "cpu")

class WaveNetLanguageModel:
    """
    A WaveNet-style language model for text generation.
    
    This implementation follows a WaveNet architecture with dilated convolutions
    and residual connections. It's designed for character-level language modeling
    using names from the dataset.
    
    Attributes:
        context_length (int): Length of input sequences for training
        generator (torch.Generator): Random number generator for reproducibility
        feature_dimension (int): Embedding dimension size
        training_batch_size (int): Number of samples per training batch
        casual_convolutional_dialation (int): Dilation factor for causal convolutions
        num_hidden_neuron (int): Number of neurons in hidden layers
        learning_iteration (int): Maximum number of training iterations
        acceptable_loss (float): Target loss threshold for early stopping
        acceptable_validation_loss (float): Target validation loss threshold
        l2_lambda (float): L2 regularization strength (weight decay coefficient)
        drop_out_layer_regularization_factor (float): Dropout probability
        show_loss_plot (bool): Whether to display loss plots during training
        learning_rate_reduction_factor (float): Factor for reducing learning rate
        learning_rate_reduction_check_interval (int): Interval for checking LR reduction
        minimum_learning_rate (float): Minimum allowed learning rate
        learning_rate (float): Initial learning rate
        words (list): List of words from the dataset
        tokenizer (Tokenizer): Tokenizer for converting text to indices
        train_loss (float): Current training loss
        validation_loss (float): Current validation loss
        test_loss (float): Current test loss
        vocabulary_size (int): Size of the vocabulary
        create_dataset (CreateDataset): Dataset creation utility
        model (nn.Sequential): The neural network model
        plotting (Plotting): Utility for plotting training progress
    """
    
    def __init__(self):
        ###########################################################################
        ################# Tweak below parameters to min-max loss ##################
        ###########################################################################

        self.context_length = 8
        self.generator = torch.Generator(device=device).manual_seed(42)
        self.feature_dimension = 12
        self.training_batch_size = 1000
        self.casual_convolutional_dialation = 2
        self.num_hidden_neuron = 64 
        self.learning_iteration = 210000
        self.acceptable_loss = 0.0

        self.acceptable_validation_loss = 2.00

        self.l2_lambda = 1e-4  # L2 regularization strength (weight decay coefficient)

        self.drop_out_layer_regularization_factor = 0.1

        self.show_loss_plot = True

        self.learning_rate_reduction_factor = 0.98 # this will be multiplied by the current learning rate upon learning rate plateau
        self.learning_rate_reduction_check_interval = 500 # after this many epochs learning rate reduction check will be performed.
        self.minimum_learning_rate = 0.001
        self.learning_rate = 0.01

        ###########################################################################
        ###########################################################################
        ###########################################################################

        self.reduce_lr_on_plateau ="reduce_lr_on_plateau"
        self.cosine_annea_ling_lr = "cosine_annea_ling_lr"
        self.one_cycle_lr = "one_cycle_lr"
        self.cosine_annealing_warm_restarts = "cosine_annealing_warm_restarts"

        self.words = open('wave-net language model/resources/names.txt', 'r').read().splitlines()
        self.tokenizer = Tokenizer(self.words)

        self.train_loss = 0.0
        self.validation_loss = 0.0
        self.test_loss = 0.0

        self.vocabulary_size = len(self.tokenizer.stoi)

        self.create_dataset = CreateDataset(device=device,
                                            tokenizer=self.tokenizer,
                                            context_lenght=self.context_length,
                                            words=self.words,
                                            generator=self.generator)
        
        self.model = nn.Sequential(
            Embedding(vocabulary_size=self.vocabulary_size,
                      generator=self.generator,
                      feature_dimension=self.feature_dimension,
                      device=device),
            
            # Layer 1: Compresses sequence from 8 to 4
            FlattenConsecutive(self.casual_convolutional_dialation, device=device), 
            nn.Linear((self.casual_convolutional_dialation * self.feature_dimension), self.num_hidden_neuron, device=device, bias=False), 
            BatchNorm1d(self.num_hidden_neuron, device=device), 
            nn.Tanh(), 
            # nn.Dropout(p=self.drop_out_layer_regularization_factor),
            ResidualBlock(self.num_hidden_neuron, self.drop_out_layer_regularization_factor, device=device),

            # Layer 2: Compresses sequence from 4 to 2
            FlattenConsecutive(self.casual_convolutional_dialation, device=device), 
            nn.Linear((self.casual_convolutional_dialation * self.num_hidden_neuron), self.num_hidden_neuron, device=device, bias=False), 
            BatchNorm1d(self.num_hidden_neuron, device=device), 
            nn.Tanh(), 
            # nn.Dropout(p=self.drop_out_layer_regularization_factor),
            ResidualBlock(self.num_hidden_neuron, self.drop_out_layer_regularization_factor, device=device),

            # Layer 3: Compresses sequence from 2 to 1
            FlattenConsecutive(self.casual_convolutional_dialation, device=device), 
            nn.Linear((self.casual_convolutional_dialation * self.num_hidden_neuron), self.num_hidden_neuron, device=device, bias=False), 
            BatchNorm1d(self.num_hidden_neuron, device=device), 
            nn.Tanh(), 
            # nn.Dropout(p=self.drop_out_layer_regularization_factor),
            ResidualBlock(self.num_hidden_neuron, self.drop_out_layer_regularization_factor, device=device),

            # Output Layer
            nn.Linear(self.num_hidden_neuron, self.vocabulary_size, device=device)
        )

        self.plotting = Plotting()
    
    def __call__(self, train_model=True, validate_model=True, test_model=True):
        """
        Execute the full training, validation and testing pipeline.
        
        Args:
            train_model (bool): Whether to train the model
            validate_model (bool): Whether to perform validation 
            test_model (bool): Whether to perform testing
            
        Returns:
            None: Results are stored in self.train_loss, self.validation_loss, and self.test_loss
        """
        n1 = int(0.8*len(self.words))
        n2 = int(0.9*len(self.words))
        training_dataset, training_target_set = self.create_dataset(self.words[:n1]) # 80%
        validation_dataset, validation_target_set = self.create_dataset(self.words[n1:n2]) #10%
        testing_dataset, testing_target_set = self.create_dataset(self.words[n2:]) #10%

        print(f"total number of parameters: {self.total_number_of_parameters()}")

        with torch.no_grad():
            validation_dataset, validation_target_set = self.create_dataset(self.words[n1:n2])
            testing_dataset, testing_target_set = self.create_dataset(self.words[n2:])

        if train_model:
            try:
                self.train_model(training_dataset=training_dataset, 
                                training_target_set=training_target_set,
                                validation_dataset=validation_dataset,
                                validation_target_set=validation_target_set)
            except KeyboardInterrupt:
                # If Ctrl+C is pressed, intercept it and print a message
                print("\n\n[!] Training interrupted by user (Ctrl+C).")
                print("--- Saving current progress, plotting, and exporting logs ---\n")

                self.perform_validation_and_test(validation_dataset=validation_dataset, 
                                         validation_target_set=validation_target_set, 
                                         testing_dataset=testing_dataset, 
                                         testing_target_set=testing_target_set, 
                                         validate_model=validate_model,
                                         test_model=test_model)
                
                self.export_run_to_csv(optimizer_name="SGD", scheduler_name="CosineAnnealingWarmRestarts")

                if self.show_loss_plot:
                    self.plotting.plot()
                return

            if self.show_loss_plot:
                self.plotting.plot()

        self.perform_validation_and_test(validation_dataset=validation_dataset, 
                                         validation_target_set=validation_target_set, 
                                         testing_dataset=testing_dataset, 
                                         testing_target_set=testing_target_set, 
                                         validate_model=validate_model,
                                         test_model=test_model)
        
        self.export_run_to_csv(optimizer_name="SGD", scheduler_name="CosineAnnealingWarmRestarts")
        
        print(f"training loss: {self.train_loss}\nvalidation loss: {self.validation_loss}\ntest loss: {self.test_loss}")

    def perform_validation_and_test(self, validation_dataset, validation_target_set, testing_dataset ,testing_target_set, validate_model=True, test_model=True):
        if validate_model:
            self.validation_loss = self.forward_pass_model(input_dataset=validation_dataset, target_dataset=validation_target_set)
        if test_model:
            self.test_loss = self.forward_pass_model(input_dataset=testing_dataset, target_dataset=testing_target_set)

    @torch.no_grad()
    def forward_pass_model(self, input_dataset, target_dataset):
        """
        Perform a forward pass through the model for validation/testing.
        
        Args:
            input_dataset (torch.Tensor): Input sequences
            target_dataset (torch.Tensor): Target sequences
            
        Returns:
            float: Loss value for the forward pass
        """
        self.model.eval() # tell the layers like BatchNorm1D to lock their state
        logits = self.model(input_dataset)

        loss = self.calculate_loss(logits=logits, target_set=target_dataset)

        self.model.train() # Switch the model back to training mode

        return loss.item()
    
    def train_model(self, training_dataset, training_target_set, validation_dataset, validation_target_set):
        """
        Train the WaveNet language model.
        
        Args:
            training_dataset (torch.Tensor): Training input sequences
            training_target_set (torch.Tensor): Training target sequences  
            validation_dataset (torch.Tensor): Validation input sequences
            validation_target_set (torch.Tensor): Validation target sequences
            
        Returns:
            None: Updates internal state with training results
        """
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)

        scheduler = self.get_learning_rate_scheduler(optimizer=optimizer, type_of_scheduler=self.cosine_annealing_warm_restarts)
        if scheduler == None:
            print(f"the learning rate is not specified. aborting training")
            return

        count = 1
        while count <= self.learning_iteration:
            #1. create training mini batch
            ix = torch.randint(0, training_dataset.shape[0], (self.training_batch_size,))

            # 2. Zero the gradients from the previous step
            optimizer.zero_grad()

            # 3. forward pass
            logits = self.model(training_dataset[ix])

            # 4. calculate loss
            loss = self.calculate_loss(logits=logits, target_set=training_target_set[ix])

            # 5. Check if reached acceptable loss
            if loss <= self.acceptable_loss:
                print(f"reached acceptable loss: {loss}")
                self.train_loss = loss.item()
                self.plotting.track(loss)
            
            # 6. backpropagation throught the layers
            loss.backward()

            # 7. Update the weights using the calculated gradients
            optimizer.step()
            
            self.train_loss = loss.item()
            self.plotting.track_train(loss) # for tracking the loss and later drawing a plot

            # 8. Check if learning rate needs reduction
            scheduler.step()

            if count % self.learning_rate_reduction_check_interval == 0:
                current_validation_loss = self.forward_pass_model(input_dataset=validation_dataset, target_dataset=validation_target_set)

                self.plotting.track_val(current_validation_loss)

                print(f"current loss: {self.train_loss} | iteration: {count}/{self.learning_iteration} | current learning rate: {optimizer.param_groups[0]['lr']} | current validation loss: {current_validation_loss}")

                if current_validation_loss <= self.acceptable_validation_loss:
                    print(f"****************** reached acceptable validation loss ******************")
                    print(f"*********************** {current_validation_loss}***********************")
                    break
            
            # 9. increment count to avoid infinite loop
            count += 1
    
    def calculate_loss(self, logits, target_set):
        """
        Calculate the cross-entropy loss with L2 regularization.
        
        Args:
            logits (torch.Tensor): Model outputs before softmax
            target_set (torch.Tensor): Target values
            
        Returns:
            torch.Tensor: Combined loss value (cross-entropy + L2 regularization)
        """
        ce_loss = F.cross_entropy(input=logits, target=target_set)

        # L2 regularization term: sum of squared weights across all parameters
        # l2_norm = sum(p.pow(2).sum() for p in self.model.parameters(recurse=True))
        # l2_penalty = (self.l2_lambda / 2) * l2_norm
        
        return ce_loss #+ l2_penalty        
    
    def total_number_of_parameters(self):
        """
        Calculate the total number of parameters in the model.
        
        Returns:
            int: Total number of trainable parameters
        """
        count = sum(p.nelement() for p in self.model.parameters(recurse=True))
        return count
    
    def get_learning_rate_scheduler(self, optimizer, type_of_scheduler="reduce_lr_on_plateau"):
        if type_of_scheduler == self.reduce_lr_on_plateau:
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, 
                mode='min', 
                factor=self.learning_rate_reduction_factor, 
                patience=3,
                min_lr=self.minimum_learning_rate,
                threshold=0.001,
                threshold_mode='abs'
            )
        elif type_of_scheduler == self.cosine_annea_ling_lr:
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer=optimizer,
                T_max=self.learning_iteration,
                eta_min=self.minimum_learning_rate
            )
        elif type_of_scheduler == self.one_cycle_lr:
            return torch.optim.lr_scheduler.OneCycleLR(
                optimizer=optimizer,
                max_lr=self.learning_rate,          # Peaks at 0.001
                total_steps=self.learning_iteration,# Total iterations (200,000)
                pct_start=0.05,                      # Spends the first 10% (10,000 steps) warming up
                div_factor=0.025,                    # Starts at max_lr / 25 (0.00004)
                final_div_factor=1.0             # Ends at starting_lr / 1000
            )
        elif type_of_scheduler == self.cosine_annealing_warm_restarts:
            return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer=optimizer,
                T_0=50000,
                eta_min=self.minimum_learning_rate
            )
        else:
            return None

    def export_run_to_csv(self, optimizer_name="SGD", scheduler_name="ReduceLROnPlateau", filename="experiment_logs.csv"):
        """
        Exports all relevant hyperparameters and final losses to a CSV file.
        Appends a new row for each run. Creates headers if the file does not exist.
        """

        file_exists = os.path.isfile(filename)

        # Compile all the data you want to track into a dictionary
        run_data = {
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Total_Parameters": self.total_number_of_parameters(),
            "Context_Length": self.context_length,
            "Feature_Dimension": self.feature_dimension,
            "Hidden_Neurons": self.num_hidden_neuron,
            "Batch_Size": self.training_batch_size,
            "Max_Iterations": self.learning_iteration,
            "Dropout_Probability": self.drop_out_layer_regularization_factor,
            "L2_Lambda": self.l2_lambda,
            "Optimizer": optimizer_name,
            "Scheduler": scheduler_name,
            "Max_Learning_Rate": self.learning_rate,
            "Min_Learning_Rate": self.minimum_learning_rate,
            "Train_Loss": round(self.train_loss, 4),
            "Validation_Loss": round(self.validation_loss, 4),
            "Test_Loss": round(self.test_loss, 4)
        }

        # Open the CSV file in 'append' mode
        with open(filename, mode='a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=run_data.keys())
            
            # Write the column headers only if this is a brand new file
            if not file_exists:
                writer.writeheader()
                
            # Write the data for the current run
            writer.writerow(run_data)
            
        print(f"\n---> Run data successfully exported to {filename} <---")