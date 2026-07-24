import matplotlib.pyplot as plt
import torch
import math

class Plotting:
    """
    A utility class for tracking and plotting training progress.
    
    This class collects training and validation losses during the training process
    and provides methods to visualize the learning curve.
    
    Attributes:
        train_losses (list): List of training loss values (logged)
        val_losses (list): List of validation loss values (logged)
        val_steps (list): List of iteration steps when validation was performed
        current_step (int): Current training step counter
    """

    def __init__(self):
        """
        Initializes the plotting utility with empty tracking lists.
        """
        self.train_losses = []
        self.val_losses = []
        self.val_steps = []  # Tracks the exact iteration for validation
        self.current_step = 0
    
    def track_train(self, loss):
        """
        Records training loss for plotting.
        
        Args:
            loss (torch.Tensor): Training loss value to record
        """
        # Track training loss and increment the step counter
        self.train_losses.append(loss.log10().item())
        self.current_step += 1
        
    def track_val(self, loss):
        """
        Records validation loss for plotting.
        
        Args:
            loss (float): Validation loss value to record
        """
        # Track validation loss and record the step it occurred on
        self.val_losses.append(math.log10(loss))
        self.val_steps.append(self.current_step)
    
    def plot(self):
        """
        Creates and displays a plot of training and validation losses.
        
        The plot shows:
        - Training loss smoothed over 1000 steps (blue line)
        - Validation loss at specific steps (red points)
        """
        # Smooth the training loss by averaging every 1000 steps
        smoothed_train = torch.tensor(self.train_losses).view(-1, 100).mean(1)
        # Create an x-axis for the smoothed training data
        train_steps = torch.arange(len(smoothed_train)) * 110
        
        # Plot Training Loss in blue
        plt.plot(train_steps, smoothed_train, color='blue', label='Training Loss')
        
        # Plot Validation Loss in red
        plt.plot(self.val_steps, self.val_losses, color='red', label='Validation Loss', marker='o')
        
        # Add labels and legend
        plt.xlabel('Iterations')
        plt.ylabel('Log10(Loss)')
        plt.legend()
        plt.show()