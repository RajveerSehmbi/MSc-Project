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
from sklearn.model_selection import train_test_split
import optuna
from sqlalchemy import create_engine
import joblib
import variables

print("Imported libraries")


# Function
# Trains a classifier using the given hyperparameters
# Includes early stopping
# Returns the trained classifier and the training and validation losses
def train_classifier(classifier, train_loader, es_loader, learning_rate, patience, device):
     
    # Cross entropy loss
    loss_function = nn.CrossEntropyLoss()
    # Optimizer
    optimizer = optim.Adam(classifier.parameters(), lr=learning_rate)
    # Early stopping
    early_stopping = EarlyStoppingClassifier(patience=patience, delta=0.005)

    # Losses
    train_losses = []
    es_losses = []


    for epoch in range(5000):

        print(f"Epoch: {epoch + 1}")

        print("Training...")
        # Training loop
        train_loss = 0
        classifier.train()
        for X, y in train_loader:
            batch_size = X.size(0)
            X = X.to(device)
            y = y.to(device)

            outputs = classifier(X)
            loss = loss_function(outputs, y)
            train_loss += loss.item() * batch_size

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)

        print(f"Training loss: {train_loss}")

        print("Early stop check...")
        # Early stopping loop
        
        classifier.eval()
        es_loss = 0.0
        for X, y in es_loader:
            with torch.no_grad():

                batch_size = X.size(0)
                X = X.to(device)
                y = y.to(device)

                outputs = classifier(X)
                loss = loss_function(outputs, y)
                es_loss += loss.item() * batch_size

        
        es_loss /= len(es_loader.dataset)
        early_stopping(es_loss, classifier)
        es_losses.append(es_loss)
        
        print(f"Early stopping loss: {es_loss}")

        if early_stopping.early_stop:
            print(f"Early stopping at epoch {epoch + 1}")
            # Load the best model
            early_stopping.load_best_weights(classifier)
            break


    
    del loss_function, optimizer, early_stopping

    return classifier, train_losses, es_losses



# Function
# Calculates the balanced accuracy of the classifier on the given dataset
# Returns the balanced accuracy
def calculate_accuracy(classifier, loader, device):

    # Create true labels
    true_labels = []
    predictions = []

    # Generate predictions
    for X, y in loader:
        
        classifier.eval()
        with torch.no_grad():
            
            true_labels.extend(y.numpy().tolist())

            X = X.to(device)

            outputs = classifier(X)
            outputs = outputs.to('cpu')
            _, predicted = torch.max(outputs, 1)
            predicted = predicted.numpy()
            predictions.extend(predicted.tolist())

    print(f"True labels: {true_labels}")
    print(f"Predictions: {predictions}")

    balanced_accuracy = balanced_accuracy_score(true_labels, predictions)

    return balanced_accuracy


def three_fold_test(X, y, testX, testy, input_dim, params, device):

    # Hyperparameters
    batch_size = params['batch_size']
    dropout_factor = params['dropout_factor']
    learning_rate = params['learning_rate']
    patience = params['patience']

    # Accuracies
    accuracies = []

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

        # Create the classifier
        # Classifier
        classifier = Classifier(input_dim, variables.pathway_num, dropout_factor).to(device)

        # Train the classifier
        classifier, tl, el = train_classifier(classifier, train_loader, val_loader, learning_rate, patience, device)

        # Calculate the validation loss
        accuracy = calculate_accuracy(classifier, test_loader, device)
        accuracies.append(accuracy)

        print(f"Accuracy: {accuracy}")

        del classifier, accuracy, tl, el
        gc.collect()

    
    print(f"Mean accuracy: {np.mean(accuracies)}")
    return np.mean(accuracies)


def final_train(X, y, input_dim, params, data_type, device):

    print("Final training...")

    # Hyperparameters
    batch_size = params['batch_size']
    dropout_factor = params['dropout_factor']
    learning_rate = params['learning_rate']
    patience = params['patience']

    # Split the data into training and validation sets
    trainX, valX, trainy, valy = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print(f"Data split. Train: {trainX.shape}, Val: {valX.shape}")

    # Datasets
    train_ds = FE_Dataset(trainX, trainy)
    val_ds = FE_Dataset(valX, valy)

    # Loaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=True, num_workers=1)

    print("Data put into loaders.")

    # Create the classifier
    # Classifier
    classifier = Classifier(input_dim, variables.pathway_num, dropout_factor).to(device)

    # Train the classifier
    classifier, train_losses, es_losses = train_classifier(classifier, train_loader, val_loader, learning_rate, patience, device)

    print("Classifier trained.")

    # Save the model
    torch.save(classifier.state_dict(), f"{variables.classifier_model_path}/classifier_{data_type}.pt")

    print("Model saved.")

    # Save the losses
    losses = pd.DataFrame({'train': train_losses, 'es': es_losses})
    losses.to_csv(f"{variables.classifier_model_path}/classifier_losses_{data_type}.csv")

    print("Losses saved.")






# ------------------------------------------------------------------------------ #
# ------------------------------------------------------------------------------ #
# ------------------------------------------------------------------------------ #
# Main code
def main(table_name):

    # Variables
    test_table_name = None
    data_type = None

    # Input dimension
    input_dim = None
    table_num = None
    if table_name == 'trainPCAtransform' or table_name == 'trainKPCAtransform':
        input_dim = variables.PCA_components
        table_num = 4
        if table_name == 'trainPCAtransform':
            test_table_name = 'testPCAtransform'
            data_type = 'PCA'
        elif table_name == 'trainKPCAtransform':
            test_table_name = 'testKPCAtransform'
            data_type = 'KPCA'
    elif table_name == 'train':
        input_dim = variables.gene_number
        table_num = 46
        test_table_name = 'test'
        data_type = 'base'

    # SQL database
    engine = create_engine(f"sqlite:///{variables.database_path}")

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

    # Test data
    testX = pd.DataFrame()
    for i in range(0, table_num):
        table = pd.read_sql(f"SELECT * FROM {test_table_name}_{i}", engine, index_col='row_num')
        testX = pd.concat([testX, table], axis=1)
        print(f"Read {test_table_name}_{i}")
    
    testy = testX['cancer_type']
    testX = testX.drop(columns=['cancer_type'])

    print("Test data read complete.")
    print(testX.shape)
    print(testy.shape)


    # Read the hyperparameters from Optuna study
    study = joblib.load(f'{variables.optuna_path_classifier}/classifier_optuna_{table_name}.pkl')
    params = study.best_params

    # Final test value
    accuracy = three_fold_test(X, y, testX, testy, input_dim, params, device)

    print(f"Final accuracy: {accuracy}")
    # Save accuracy in text file
    with open(f"{variables.classifier_model_path}/classifier_accuracy_{data_type}.txt", 'w') as f:
        f.write(f"{accuracy}")
    
    print("Accuracy saved.")

    # Train the final model
    final_train(X, y, input_dim, params, data_type, device)

    print("Final training complete.")



if __name__ == "__main__":

    # Read the inputs
    table_name = sys.argv[1]

    main(table_name)
    print("Optuna complete.")