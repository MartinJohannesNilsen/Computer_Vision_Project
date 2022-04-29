import numpy as np
import torch

# Inspiration from https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py
# Removed some of the code
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, delta=0,):
        """
        Args:
            patience (int): How long to wait after last time mAP improved.
                            Default: 7
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0          
        """
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
    def __call__(self, mAP):
        """
        Method to be used for every epoch
            Parameters:
            mAP: mAP@0.5:0.95 score for each epoch
            
            Returns:
            boolean: True if epoch is best score
        """
    
        score = mAP

        if self.best_score is None:
            self.best_score = score
            print(f"New best score: {score}")
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            print(f"New best score: {score}")
            self.best_score = score
            self.counter = 0
            return True
            

