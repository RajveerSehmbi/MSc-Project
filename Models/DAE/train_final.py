import sys
sys.path.append('/vol/bitbucket/rs218/MSc-Project')

from torch.utils.data import DataLoader
from Models.datasets import FE_Dataset
from deepDAE import DeepDAE
from Models.early_stop import EarlyStoppingAE
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import torch
import optuna
import joblib
import math
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
import variables

import gc
import sys

print("Libraries imported.")

def train(autoencoder, device, train_loader, val_loader, learning_rate, patience):

    # Loss function
    loss_function = nn.MSELoss()
    # Optimizer
    optimizer = optim.Adam(autoencoder.parameters(), learning_rate)
    # Early stopping
    early_stopping = EarlyStoppingAE(patience=patience, delta=0.01)

    # Losses
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

        train_loss /= train_loader.dataset.__len__()
        train_loss = math.sqrt(train_loss)
        train_losses.append(train_loss)
        print(f"Training RMSE loss: {train_loss}")
        
        print("Early stop check...")
        # Early stopping loop
        autoencoder.eval()
        es_loss = 0.0
        for X, y in val_loader:
            with torch.no_grad():
                batch_size = X.size(0)

                X = X.to(device)
                reconstructed = autoencoder.val_forward(X)
                if autoencoder.split:
                    reconstructed = reconstructed.to('cuda:0')
                loss = loss_function(reconstructed, X)
                es_loss += (loss.item() * batch_size)


        es_loss /= val_loader.dataset.__len__()
        es_loss = math.sqrt(es_loss)
        es_losses.append(es_loss)
        early_stopping(es_loss, autoencoder)
        print(f"Early stopping RMSE loss: {es_loss}")

        if early_stopping.early_stop:
            print(f"Early stopping at epoch {epoch+1}.")
            early_stopping.load_best_weights(autoencoder)
            break

    return autoencoder, train_losses, es_losses



def calculate_loss(autoencoder, loader, device):
    
        loss_function = nn.MSELoss()
        total_loss = 0.0
        autoencoder.eval()
        for X, y in loader:
            with torch.no_grad():
                batch_size = X.size(0)
    
                X = X.to(device)
                reconstructed = autoencoder.val_forward(X)
                if autoencoder.split:
                    reconstructed = reconstructed.to('cuda:0')
                loss = loss_function(reconstructed, X)
                total_loss += (loss.item() * batch_size)


        total_loss /= loader.dataset.__len__()
        total_loss = math.sqrt(total_loss)
        return total_loss


def three_fold_test(X, y, testX, testy, gene_order, params, model_type):

    # Hyperparameters
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    noise_type = None
    pathway_proportion = None
    if variables.DAE_type == 'standard':
        noise_type = params['params_noise_type'].iloc[0]
        pathway_proportion = 0.1 # Not used in standard DAE
    elif variables.DAE_type == 'pathway':
        noise_type = 'pathway'
        pathway_proportion = params['params_pathway_proportion'].iloc[0]
    noise_factor = params['params_noise_factor'].iloc[0]
    dropout_rate = params['params_dropout_rate'].iloc[0]
    batch_size = params['params_batch_size'].iloc[0]
    learning_rate = params['params_learning_rate'].iloc[0]
    patience = params['params_patience'].iloc[0]

    # Ensure correct data type
    pathway_proportion = float(pathway_proportion)
    noise_factor = float(noise_factor)
    dropout_rate = float(dropout_rate)
    batch_size = int(batch_size)
    learning_rate = float(learning_rate)
    patience = int(patience)

    # Accuracies
    losses = []

    # Test ds
    test_ds = FE_Dataset(testX, testy)
    test_loader = DataLoader(test_ds, batch_size=batch_size, num_workers=1)

    random_states = [42, 43, 44]

    for i in range(0, 3):
        print(f"Fold {i}")

        # Split the data
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=random_states[i], stratify=y)

        print(f"Train: {X_train.shape}, Val: {X_val.shape}")
        print(f"Train: {y_train.shape}, Val: {y_val.shape}")

        train_ds = FE_Dataset(X_train, y_train)
        val_ds = FE_Dataset(X_val, y_val)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=True, num_workers=1)

        # Create the autoencoder
        autoencoder = DeepDAE(noise_factor, noise_type, dropout_rate, device, gene_order, pathway_proportion, split=True)

        # Train the autoencoder
        autoencoder, tl, el = train(autoencoder, device, train_loader, val_loader, learning_rate, patience)

        if random_states[i] == 42:
            # Save encoder and decoder separately
            torch.save(autoencoder.encoder.state_dict(), f"{variables.DAE_model_path}/DAE_encoder_{model_type}.pt")
            torch.save(autoencoder.decoder.state_dict(), f"{variables.DAE_model_path}/DAE_decoder_{model_type}.pt")
            print("Encoder and decoder saved.")

            # Save the losses
            losses = pd.DataFrame({'train': tl, 'es': el})
            losses.to_csv(f"{variables.DAE_model_path}/DAE_losses_{model_type}.csv")

            print("Losses saved.")

        # Calculate the validation loss
        loss = calculate_loss(autoencoder, test_loader, device)
        losses.append(loss)

        print(f"RMSE Loss: {loss}")

        del autoencoder, loss, tl, el
        gc.collect()

    
    print(f"Mean RMSE loss: {np.mean(losses)}")
    return np.mean(losses)






 # Main function
def main():


    print("Loading data...")
    engine = create_engine(f"sqlite:///{variables.database_path}")
    data = pd.DataFrame()

    for i in range(0, 46):
        table = pd.read_sql(f"SELECT * FROM train_{i}", engine, index_col='row_num')
        data = pd.concat([data, table], axis=1)
        print(f"Read train_{i}")

    print(f"Train Data read with shape: {data.shape}")

    X = data.drop(columns=['cancer_type'])
    y = data['cancer_type']

    del data
    gc.collect()

    testX = pd.DataFrame()
    for i in range(0, 46):
        table = pd.read_sql(f"SELECT * FROM test_{i}", engine, index_col='row_num')
        testX = pd.concat([testX, table], axis=1)
        print(f"Read test_{i}")

    print(f"Test Data read with shape: {testX.shape}")
    
    testy = testX['cancer_type']
    testX = testX.drop(columns=['cancer_type'])


    gene_order = X.columns
    gene_order = [gene.split('.')[0] for gene in gene_order]
    print("Gene order created.")

    best_params = None
    
    if variables.DAE_type == 'standard':
        best_params = pd.read_csv(f"{variables.optuna_path}/deepDAE_best_params.csv")
    elif variables.DAE_type == 'pathway':
        best_params = pd.read_csv(f"{variables.optuna_path}/PWdeepDAE_best_params.csv")

    print("Best params loaded.")

    # Test the model
    print("Testing the model...")

    type = None
    if variables.DAE_type == 'standard':
        type = 'deepDAE'
    elif variables.DAE_type == 'pathway':
        type = 'PWdeepDAE'

    loss = three_fold_test(X, y, testX, testy, gene_order, best_params, type)

    print(f"Final RMSE loss: {loss}")
    # Save loss in text file
    with open(f"{variables.DAE_model_path}/DAE_loss_{type}.txt", 'w') as f:
        f.write(f"{loss}")

    print("Loss saved.")





if __name__ == "__main__":
    main()
