import sys
sys.path.append('/vol/bitbucket/rs218/MSc-Project')

from nn import Classifier
print("Imported Classifier")

from torch.utils.data import DataLoader
from Models.datasets import FE_Dataset
import torch
import torch.nn as nn
import torch.optim as optim
import gc
from Models.early_stop import EarlyStoppingClassifier
import pandas as pd
import numpy as np
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
import optuna
from sqlalchemy import create_engine
import joblib
import variables

print("Imported libraries")


# Function
# Trains a classifier using the given hyperparameters
# Includes early stopping
# Returns the trained classifier and the training and validation losses
def train_classifier(classifier, train_loader, es_loader, learning_rate, patience):
     
    # Cross entropy loss
    loss_function = nn.CrossEntropyLoss()
    # Optimizer
    optimizer = optim.Adam(classifier.parameters(), lr=learning_rate)
    # Early stopping
    early_stopping = EarlyStoppingClassifier(patience=patience, delta=0.005)


    for epoch in range(5000):

        print(f"Epoch: {epoch + 1}")

        print("Training...")
        # Training loop
        train_loss = 0
        classifier.train()
        for X, y in train_loader:
            batch_size = X.size(0)

            outputs = classifier(X)
            loss = loss_function(outputs, y)
            train_loss += loss.item() * batch_size

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_loss /= len(train_loader.dataset)

        print(f"Training loss: {train_loss}")

        print("Early stop check...")
        # Early stopping loop
        
        classifier.eval()
        es_loss = 0.0
        for X, y in es_loader:
            with torch.no_grad():

                batch_size = X.size(0)

                outputs = classifier(X)
                loss = loss_function(outputs, y)
                es_loss += loss.item() * batch_size

        
        es_loss /= len(es_loader.dataset)
        early_stopping(es_loss, classifier)
        
        print(f"Early stopping loss: {es_loss}")

        if early_stopping.early_stop:
            print(f"Early stopping at epoch {epoch + 1}")
            # Load the best model
            early_stopping.load_best_weights(classifier)
            break


    
    del loss_function, optimizer, early_stopping

    return classifier



# Function
# Calculates the balanced accuracy of the classifier on the given dataset
# Returns the balanced accuracy
def calculate_accuracy(classifier, loader):

    # Create true labels
    true_labels = []
    predictions = []

    # Generate predictions
    for X, y in loader:
        
        classifier.eval()
        with torch.no_grad():
            
            true_labels.extend(y.numpy().tolist())

            outputs = classifier(X)
            _, predicted = torch.max(outputs, 1)
            predicted = predicted.numpy()
            predictions.extend(predicted.tolist())

    balanced_accuracy = balanced_accuracy_score(true_labels, predictions)

    return balanced_accuracy





def full_train(trial, X, y, input_dim):

    # Hyperparameters
    batch_sizes = [512, 256, 128, 64, 32, 16]
    batch_size = trial.suggest_categorical('batch_size', batch_sizes)
    dropout_factor = trial.suggest_float('dropout_factor', 0.0, 0.5, step=0.1)
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    patience = trial.suggest_int('patience', 5, 20)

    # KFold
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    # Accuracies
    accuracies = []

    count = 0
    for train_index, val_index in skf.split(X, y):

        print(f"New fold {count}")
        print(f"Train: {X.iloc[train_index].shape}, Val: {X.iloc[val_index].shape}")

        train_ds = FE_Dataset(X.iloc[train_index], y.iloc[train_index])
        val_ds = FE_Dataset(X.iloc[val_index], y.iloc[val_index])

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=True, num_workers=1)

        # Create the classifier
        # Classifier
        classifier = Classifier(input_dim, variables.pathway_num, dropout_factor)

        # Train the classifier
        classifier = train_classifier(classifier, train_loader, val_loader, learning_rate, patience)

        # Calculate the validation loss
        accuracy = calculate_accuracy(classifier, val_loader)
        accuracies.append(accuracy)

        print(f"Accuracy: {accuracy}")

        count += 1

        del classifier, accuracy
        gc.collect()
    
    print(f"Mean accuracy: {np.mean(accuracies)}")
    return np.mean(accuracies)




# ------------------------------------------------------------------------------ #
# ------------------------------------------------------------------------------ #
# ------------------------------------------------------------------------------ #
# Main code
def main(table_name):

    # Input dimension
    input_dim = None
    table_num = None
    if table_name == 'trainPCAtransform' or table_name == 'trainPWdeepDAEtransformed' or table_name == 'trainKPCAtransform' or table_name == 'traindeepDAEtransformed':
        input_dim = variables.PCA_components
        table_num = 4
    elif table_name == 'train':
        input_dim = variables.gene_number
        table_num = 46

    engine = create_engine(f"sqlite:///{variables.database_path}")
    X = pd.DataFrame()

    for i in range(0, table_num):
        table = pd.read_sql(f"SELECT * FROM {table_name}_{i}", engine, index_col='row_num')
        X = pd.concat([X, table], axis=1)
        print(f"Read {table_name}_{i}")
    
    y = X['cancer_type']
    X = X.drop(columns=['cancer_type'])

    print("Data read complete.")
    print(X.shape)
    print(y.shape)

    sampler = optuna.samplers.TPESampler()
      
    study = optuna.create_study(sampler=sampler, direction='maximize')
    study.optimize(lambda trial: full_train(trial, X, y, input_dim), n_trials=10)

    joblib.dump(study, f'{variables.optuna_path_classifier}/classifier_optuna_{table_name}.pkl')



if __name__ == "__main__":

    # Read the inputs
    table_name = sys.argv[1]

    main(table_name)
    print("Optuna complete.")