import matplotlib.pyplot as plt
import torch

class Plotting:

    def __init__(self):
        self.lossi = []
    
    def track(self, loss):
        self.lossi.append(loss.log10().item())
    
    def plot(self):
        plt.plot(torch.tensor(self.lossi).view(-1, 1000).mean(1))
        plt.show()