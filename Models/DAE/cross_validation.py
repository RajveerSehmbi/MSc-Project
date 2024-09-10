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
from sklearn.model_selection import StratifiedKFold
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


        val_loss /= val_loader.dataset.__len__()
        val_loss = math.sqrt(val_loss)
        return val_loss




def cross_val(X, y, gene_order, params):

    print(params)

    # Accuracies
    cv_accuracies = []

    for i in range(0, 3):

        print(f"Param set {i}")

        # Hyperparameters, taken from params dataframe for the ith row
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        noise_type = None
        pathway_proportion = None
        if variables.DAE_type == 'standard':
            noise_type = params['params_noise_type'].iloc[i]
            pathway_proportion = 0.1 # Not used in standard DAE
        elif variables.DAE_type == 'pathway':
            noise_type = 'pathway'
            pathway_proportion = params['params_pathway_proportion'].iloc[i]
        noise_factor = params['params_noise_factor'].iloc[i]
        dropout_rate = params['params_dropout_rate'].iloc[i]
        batch_size = params['params_batch_size'].iloc[i]
        learning_rate = params['params_learning_rate'].iloc[i]
        patience = params['params_patience'].iloc[i]

        # Ensure correct data type
        pathway_proportion = float(pathway_proportion)
        noise_factor = float(noise_factor)
        dropout_rate = float(dropout_rate)
        batch_size = int(batch_size)
        learning_rate = float(learning_rate)
        patience = int(patience)

        # Accuracies
        accuracies = []

        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

        count = 1
        for train_index, val_index in skf.split(X, y):
            print(f"Fold {count}")

            # Split the data
            print(f"Train: {X.iloc[train_index].shape}, Val: {X.iloc[val_index].shape}")

            train_ds = FE_Dataset(X.iloc[train_index], y.iloc[train_index])
            val_ds = FE_Dataset(X.iloc[val_index], y.iloc[val_index])

            train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
            val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=True, num_workers=1)

            # Create the autoencoder
            autoencoder = DeepDAE(noise_factor, noise_type, dropout_rate, device, gene_order, pathway_proportion, split=True)

            # Train the autoencoder
            print("Training...")
            autoencoder, tl, el = train(autoencoder, device, train_loader, val_loader, learning_rate, patience)
            print("Autoencoder trained.")

            # Calculate the validation loss
            val_loss = calculate_val_loss(autoencoder, device, val_loader)
            print(f"Validation RMSE loss: {val_loss}")

            del autoencoder, train_loader, val_loader, tl, el
            gc.collect()
            torch.cuda.empty_cache()

            accuracies.append(val_loss)

        print(f"Mean accuracy: {np.mean(accuracies)}")
        cv_accuracies.append(np.mean(accuracies))

    # Add the cv accuracies to the params
    params['cv_accuracy'] = cv_accuracies

    # Save the params
    if variables.DAE_type == 'standard':
        pd.DataFrame(params).to_csv(f"{variables.optuna_path}/deepDAE_cvparams.csv", index=False)
    elif variables.DAE_type == 'pathway':
        pd.DataFrame(params).to_csv(f"{variables.optuna_path}/PWdeepDAE_cvparams.csv", index=False)

    # Return the index of the best accuracy
    return np.argmin(cv_accuracies)



 # Main function
def main():


    print("Loading data...")
    engine = create_engine(f"sqlite:///{variables.database_path}")
    data = pd.DataFrame()

    for i in range(0, 46):
        table = pd.read_sql(f"SELECT * FROM train_{i}", engine, index_col='row_num')
        data = pd.concat([data, table], axis=1)
        print(f"Read train_{i}")

    print(f"Data read with shape: {data.shape}")

    X = data.drop(columns=['cancer_type'])
    y = data['cancer_type']

    del data
    gc.collect()


    gene_order = X.columns
    gene_order = [gene.split('.')[0] for gene in gene_order]
    print("Gene order created.")


    top3_params = None
    
    if variables.DAE_type == 'standard':
        top3_params = pd.read_csv(f"{variables.optuna_path}/deepDAE_top3_params.csv")
    elif variables.DAE_type == 'pathway':
        top3_params = pd.read_csv(f"{variables.optuna_path}/PWdeepDAE_top3_params.csv")

    print("Optuna study loaded.")

    # Get the index of the best param set
    best_index = cross_val(X, y, gene_order, top3_params)

    print(f"Best index: {best_index}")

    # Get the best params
    best_params = top3_params.iloc[best_index].to_dict()

    print("Best params:")
    print(best_params)

    # Save best params to a file
    if variables.DAE_type == 'standard':
        pd.DataFrame(best_params, index=[0]).to_csv(f"{variables.optuna_path}/deepDAE_best_params.csv", index=False)
    elif variables.DAE_type == 'pathway':
        pd.DataFrame(best_params, index=[0]).to_csv(f"{variables.optuna_path}/PWdeepDAE_best_params.csv", index=False)



if __name__ == "__main__":
    main()
