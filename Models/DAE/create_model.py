import sys
sys.path.append('/vol/bitbucket/rs218/MSc-Project')

from torch.utils.data import DataLoader
from datasets import FE_Dataset
from deepDAE import DeepSDAE
from early_stop import EarlyStoppingAE
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import torch
import math
import variables

import gc

print("Libraries imported.")

def train(autoencoder, device, train_loader, es_loader, learning_rate):

    # Loss function
    loss_function = nn.MSELoss()
    # Optimizer
    optimizer = optim.Adam(autoencoder.parameters(), learning_rate)
    # Early stopping
    early_stopping = EarlyStoppingAE(patience=10, delta=0.005)

    train_losses = []
    es_losses = []

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
        train_losses.append(train_loss)
        print(f"Training loss: {train_loss}")
        
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
        es_losses.append(es_loss)
        early_stopping(es_loss, autoencoder)
        print(f"Early stopping loss: {es_loss}")

        if early_stopping.early_stop:
            print(f"Early stopping at epoch {epoch+1}.")
            early_stopping.load_best_weights(autoencoder)
            break

    return autoencoder, train_losses, es_losses


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



def full_train(train_set, es_set, val_set, test_set):

    # Hyperparameters
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    noise_type = 'gaussian'
    noise_factor = 0.1
    dropout_rate = 0.1
    batch_size = 128
    learning_rate = 0.000369

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=3)
    es_loader = DataLoader(es_set, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=1)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=1)

    # Create the autoencoder
    autoencoder = DeepSDAE(noise_factor, noise_type, dropout_rate, device, split=True)

    # Train the autoencoder
    print("Training...")
    print(f"Batch size: {batch_size}, noise type: {noise_type}, noise factor: {noise_factor}, dropout rate: {dropout_rate}, learning rate: {learning_rate}")
    autoencoder, train_losses, es_losses = train(autoencoder, device, train_loader, es_loader, learning_rate)
    print("Autoencoder trained.")

    val_loss = calculate_val_loss(autoencoder, device, val_loader)
    print(f"Validation RMSE loss: {val_loss}")

    test_loss = calculate_val_loss(autoencoder, device, test_loader)
    print(f"Test RMSE loss: {test_loss}")

    # Save the model
    torch.save(autoencoder.state_dict(), f"{variables.DAE_model_path}/{variables.DAE_type}_model.pt")
    print("Model saved.")

    del autoencoder
    gc.collect()
    torch.cuda.empty_cache()

    # Save validation and test losses
    losses = pd.DataFrame({'val': val_loss, 'test': test_loss}, index=[0])
    losses.to_csv(f"{variables.DAE_model_path}/{variables.DAE_type}_model_final_losses.csv")

    # Put the losses in a dataframe and save them to a file
    losses = pd.DataFrame({'train': train_losses, 'es': es_losses})
    losses.to_csv("{variables.DAE_model_path}/{variables.DAE_type}_model_train_losses.csv")
    print("Losses saved.")

    del losses, train_losses, es_losses
    gc.collect()



 # Main function
def main():

    print("Loading data...")

    # Load the data
    train_set = FE_Dataset('train')
    es_set = FE_Dataset('es')
    val_set = FE_Dataset('val')
    test_set = FE_Dataset('test')
    print("Data loaded.")

    full_train(train_set, es_set, val_set, test_set)
    print("Training complete.")




if __name__ == "__main__":
    main()
