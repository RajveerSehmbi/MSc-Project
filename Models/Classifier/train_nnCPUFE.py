import sys
sys.path.append('/vol/bitbucket/rs218/MSc-Project')

from nn import Classifier
print("Imported Classifier")
from Models.datasets import FE_Dataset
print("Imported Dataset")
from torch.utils.data import DataLoader

import torch
import torch.nn as nn
import torch.optim as optim
import gc
from Models.early_stop import EarlyStoppingClassifier
import pandas as pd
import numpy as np
from sklearn.metrics import balanced_accuracy_score
import variables

print("Imported libraries")

# 6005
# 24017


# Function
# Trains a classifier using the given hyperparameters
# Includes early stopping
# Returns the trained classifier and the training and validation losses
def train_classifier(classifier, train_loader, es_loader, learning_rate):
     
    # Cross entropy loss
    loss_function = nn.CrossEntropyLoss()
    # Optimizer
    optimizer = optim.Adam(classifier.parameters(), lr=learning_rate)
    # Early stopping
    early_stopping = EarlyStoppingClassifier(patience=10, delta=0.005)

    # Keep losses
    train_losses = []
    val_losses = []

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

        train_loss /= 24017
        train_losses.append(train_loss)

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

        
        es_loss /= 6005
        val_losses.append(es_loss)
        early_stopping(es_loss, classifier)
        
        print(f"Early stopping loss: {es_loss}")

        if early_stopping.early_stop:
            print(f"Early stopping at epoch {epoch + 1}")
            # Load the best model
            early_stopping.load_best_weights(classifier)
            break


    
    del loss_function, optimizer, early_stopping, train_loss, es_loss
    gc.collect()

    return classifier, train_losses, val_losses



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



# Function
# Performs a hyperparameter search to find the best hyperparameters for the classifier
# Explores the following hyperparameters:
# - Input dimension
# - Hidden dimension
# - Batch size
# - Dropout factor
# Saves the best hyperparameters and the validation accuracy for each classifier
# Returns a data frame of the best hyperparameters and validation accuracies
def hyperparameter_search():

    print("Starting hyperparameter search...")
    
    # Hyperparameters
    dimensions = {variables.PCA_components: (variables.PCA_components // 2)}
    batch_sizes = [512, 256, 128, 64, 32, 16]
    dropout_factors = [0.2, 0.3, 0.4, 0.5]
    learning_rates = [0.01, 0.001, 0.0001]

    # Store the best hyperparameters
    best_hyperparameters = {variables.PCA_components: None}
    val_accuracies = {variables.PCA_components: None}

    # Save training losses
    train_loss_list = None
    es_loss_list = None

    # Loop over the hyperparameters
    for input_dim, hidden_dim in dimensions.items():
        best_accuracy = 0.0

        train_ds = FE_Dataset('trainPCAtransform')
        es_ds = FE_Dataset('esPCAtransform')
        val_ds = FE_Dataset('valPCAtransform')

        for batch_size in batch_sizes:

            train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
            es_loader = DataLoader(es_ds, batch_size=batch_size, shuffle=True, num_workers=1)
            val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=True, num_workers=1)

            for dropout_factor in dropout_factors:
                for learning_rate in learning_rates:

                    print(f"Training autoencoder with input dimension: {input_dim}, \
                            hidden dimension: {hidden_dim}, \
                            batch size: {batch_size}, \
                            dropout factor: {dropout_factor} \
                            learning rate: {learning_rate}")

                    classifier = Classifier(input_dim, hidden_dim, dropout_factor)
                    print("Classifier created.")

                    # Train the classifier
                    classifier, train_losses, es_losses = train_classifier(classifier, train_loader, es_loader, learning_rate)
                    print("Classifier trained.")

                    # Calculate the validation loss
                    accuracy = calculate_accuracy(classifier, val_loader)
                    print(f"Validation accuracy: {accuracy}")

                    # Save the best hyperparameters
                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        best_hyperparameters[input_dim] = [batch_size, dropout_factor, learning_rate]
                        val_accuracies[input_dim] = accuracy
                        train_loss_list = train_losses
                        es_loss_list = es_losses
                        # Save the model
                        torch.save(classifier.state_dict(), f"{variables.classifier_model_path}/PCAFEclassifier_{input_dim}_{hidden_dim}.pt")
                        print("Model saved.")

                    del classifier, train_losses, es_losses, accuracy
                    gc.collect()

                    print(f"Best Validation accuracy: {best_accuracy}")

        # Create a data frame of the losses
        losses = pd.DataFrame({'train_loss': train_loss_list, 'es_loss': es_loss_list})
        losses.to_csv(f"{variables.classifier_model_path}/PCAFElosses_{input_dim}_{hidden_dim}.csv")

        del losses
        gc.collect()

    # Save info for each classifier
    classifier_info = pd.DataFrame(columns=['input_dim', 'hidden_dim', 'best_batch_size', 'best_dropout_factor', 'best_learning_rate', 'val_accuracy'])
    for key, value in best_hyperparameters.items():
        row = pd.DataFrame([{
            'input_dim': key,
            'hidden_dim': dimensions[key],
            'best_batch_size': value[0],
            'best_dropout_factor': value[1],
            'best_learning_rate': value[2],
            'val_accuracy': val_accuracies[key]
        }])
        classifier_info = pd.concat([classifier_info, row], ignore_index=True)

    return classifier_info



def evaluate_on_test(parameters):

    test_accuracies = []

    for index, row in parameters.iterrows():
        input_dim = row['input_dim']
        hidden_dim = row['hidden_dim']
        batch_size = row['best_batch_size']
        dropout_factor = row['best_dropout_factor']
        learning_rate = row['best_learning_rate']

        print(f"Evaluating classifier with input dimension: {input_dim}, \
                hidden dimension: {hidden_dim}, \
                batch size: {batch_size}, \
                dropout factor: {dropout_factor} \
                learning rate: {learning_rate}")

        classifier = Classifier(input_dim, hidden_dim, dropout_factor)
        classifier.load_state_dict(torch.load(f"{variables.classifier_model_path}/PCAFEclassifier_{input_dim}_{hidden_dim}.pt"))

        test_ds = FE_Dataset('testPCAtransform')
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=True, num_workers=1)

        accuracy = calculate_accuracy(classifier, test_loader)
        test_accuracies.append(accuracy)

        del classifier, accuracy
        gc.collect()

    parameters['test_accuracy'] = test_accuracies
    parameters.to_csv(f"{variables.classifier_model_path}/PCAFEclassifier_info.csv")



# ------------------------------------------------------------------------------ #
# ------------------------------------------------------------------------------ #
# ------------------------------------------------------------------------------ #
# Main code

if __name__ == "__main__":


    classifier_parameters = hyperparameter_search()
    print("Finished hyperparameter search.")

    evaluate_on_test(classifier_parameters)
    print("Finished evaluating on test set.")