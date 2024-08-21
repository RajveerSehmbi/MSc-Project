import sys
sys.path.append('/vol/bitbucket/rs218/Project')

import torch
import os
import variables


# FOR USE IN AUTOENCODERS
class EarlyStoppingAE:
    def __init__(self, patience=10, delta=0):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.early_stop = False
        self.counter = 0

    def __call__(self, val_loss, model):
        score = val_loss

        if self.best_score is None:
            self.best_score = score
            torch.save(model.state_dict(), f"{variables.es_path}/deepdae_{model.noiserate}_{model.noise_type}_{model.dropout_factor}.pt")
        elif score > self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            # Save the model weights in a file
            torch.save(model.state_dict(), f"{variables.es_path}/deepdae_{model.noiserate}_{model.noise_type}_{model.dropout_factor}.pt")
            self.counter = 0

    def load_best_weights(self, model):
        model.load_state_dict(torch.load(f"{variables.es_path}/deepdae_{model.noiserate}_{model.noise_type}_{model.dropout_factor}.pt"))
        # Delete the file
        os.remove(f"{variables.es_path}/deepdae_{model.noiserate}_{model.noise_type}_{model.dropout_factor}.pt")



# FOR USE IN CLASSIFICATION
class EarlyStoppingClassifier:
    def __init__(self, patience=10, delta=0):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.early_stop = False
        self.counter = 0

    def __call__(self, val_loss):
        score = val_loss

        if self.best_score is None:
            self.best_score = score
        elif score > self.best_score - self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
