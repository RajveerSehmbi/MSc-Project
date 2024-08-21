import sys
sys.path.append('/vol/bitbucket/rs218/Project')

from torch.utils.data import DataLoader
from Models.datasets import FE_Dataset
from deepDAE import DeepSDAE
from Models.early_stop import EarlyStoppingAE
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import torch
import optuna
import joblib
import math
import variables

import gc
import sys

print("Libraries imported.")

def train(autoencoder, device, train_loader, es_loader, learning_rate):

    # Loss function
    loss_function = nn.MSELoss()
    # Optimizer
    optimizer = optim.Adam(autoencoder.parameters(), learning_rate)
    # Early stopping
    early_stopping = EarlyStoppingAE(patience=10, delta=0.005)

    for epoch in range(5000):
        print(f"Epoch: {epoch + 1}")

        print("Training...")
        # Training loop
        train_loss = 0
        autoencoder.train()
        for X, y in train_loader:
            batch_size = X.size(0)

            X = X.to(device)
            reconstructed = autoencoder(X)
            if autoencoder.split:
                reconstructed = reconstructed.to('cuda:0')
            loss = loss_function(reconstructed, X)
            train_loss += (loss.item() * batch_size)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        gc.collect()
        torch.cuda.empty_cache()

        train_loss /= 24017
        train_loss = math.sqrt(train_loss)
        print(f"Training RMSE loss: {train_loss}")
        
        print("Early stop check...")
        # Early stopping loop
        autoencoder.eval()
        es_loss = 0.0
        for X, y in es_loader:
            with torch.no_grad():
                batch_size = X.size(0)

                X = X.to(device)
                reconstructed = autoencoder.val_forward(X)
                if autoencoder.split:
                    reconstructed = reconstructed.to('cuda:0')
                loss = loss_function(reconstructed, X)
                es_loss += (loss.item() * batch_size)

        gc.collect()
        torch.cuda.empty_cache()

        es_loss /= 6005
        es_loss = math.sqrt(es_loss)
        early_stopping(es_loss, autoencoder)
        print(f"Early stopping RMSE loss: {es_loss}")

        if early_stopping.early_stop:
            print(f"Early stopping at epoch {epoch+1}.")
            early_stopping.load_best_weights(autoencoder)
            break

    return autoencoder


def calculate_val_loss(autoencoder, device, val_loader):
    
        loss_function = nn.MSELoss()
        val_loss = 0.0
        autoencoder.eval()
        for X, y in val_loader:
            with torch.no_grad():
                batch_size = X.size(0)
    
                X = X.to(device)
                reconstructed = autoencoder.val_forward(X)
                if autoencoder.split:
                    reconstructed = reconstructed.to('cuda:0')
                loss = loss_function(reconstructed, X)
                val_loss += (loss.item() * batch_size)

        gc.collect()
        torch.cuda.empty_cache()

        val_loss /= 1106
        val_loss = math.sqrt(val_loss)
        return val_loss



def full_train(trial, train_set, es_set, val_set, gene_order, DAE_type):

    # Hyperparameters
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    noise_type = None
    if DAE_type == 'standard':
        noise_type = trial.suggest_categorical('noise_type', ['gaussian', 'masking'])
    elif DAE_type == 'pathway':
        noise_type = 'pathway'
    noise_factor = trial.suggest_float('noise_factor', 0.1, 0.5, step=0.1)
    dropout_rate = trial.suggest_float('dropout_rate', 0.0, 0.5, step=0.1)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128, 256, 512])
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    pathway_proportion = trial.suggest_float('pathway_proportion', 0.0, 1.0, step=0.1)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=3)
    es_loader = DataLoader(es_set, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=1)

    # Create the autoencoder
    autoencoder = DeepSDAE(noise_factor, noise_type, dropout_rate, device, gene_order, pathway_proportion, split=True)

    # Train the autoencoder
    print("Training...")
    print(f"Batch size: {batch_size}, pathway proportion: {pathway_proportion}, noise type: {noise_type}, noise factor: {noise_factor}, dropout rate: {dropout_rate}, learning rate: {learning_rate}")
    autoencoder = train(autoencoder, device, train_loader, es_loader, learning_rate)
    print("Autoencoder trained.")

    # Calculate the validation loss
    val_loss = calculate_val_loss(autoencoder, device, val_loader)
    print(f"Validation RMSE loss: {val_loss}")

    return val_loss



 # Main function
def main():

    gene_order = pd.read_csv(f'{variables.gene_order_file}')
    gene_order = gene_order['0']
    gene_order = [gene.split('.')[0] for gene in gene_order]

    print("Gene order loaded")

    print("Loading data...")

    # Load the data
    train_set = FE_Dataset('train')
    es_set = FE_Dataset('es')
    val_set = FE_Dataset('val')

    print("Data loaded.")

    sampler = optuna.samplers.TPESampler()
      
    study = optuna.create_study(sampler=sampler, direction='minimize')
    study.optimize(lambda trial: full_train(trial, train_set, es_set, val_set, gene_order, variables.DAE_type), n_trials=20)

    if variables.DAE_type == 'standard':
        joblib.dump(study, f'{variables.optuna_path}/deepDAE_optuna.pkl')
    elif variables.DAE_type == 'pathway':
        joblib.dump(study, f'{variables.optuna_path}/PWdeepSDAE_optuna.pkl')

    print("Optuna study saved.")


if __name__ == "__main__":
    main()
