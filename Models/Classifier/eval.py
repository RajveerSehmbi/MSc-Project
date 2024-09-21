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
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, balanced_accuracy_score
from sklearn.model_selection import train_test_split
import optuna
from sqlalchemy import create_engine
import joblib
import variables

def evaluate(classifier, loader, data_type):

    # Create true labels
    true_labels = []
    predictions = []

    # Generate predictions
    for X, y in loader:
        
        classifier.eval()
        with torch.no_grad():
            
            true_labels.extend(y.numpy().tolist())

            outputs = classifier(X)
            outputs = outputs.to('cpu')
            _, predicted = torch.max(outputs, 1)
            predicted = predicted.numpy()
            predictions.extend(predicted.tolist())

    print(f"True labels: {true_labels}")
    print(f"Predictions: {predictions}")

    balanced_accuracy = balanced_accuracy_score(true_labels, predictions)
    conf_matrix = confusion_matrix(true_labels, predictions)
    precision = precision_score(true_labels, predictions, average='weighted')  # or 'macro'
    recall = recall_score(true_labels, predictions, average='weighted')  # or 'macro'
    f1 = f1_score(true_labels, predictions, average='weighted')  # or 'macro'

    # Save all metrics into a file
    with open(f"{variables.classifier_model_path}/{data_type}_metrics.txt", "a") as f:
        f.write(f"Balanced accuracy: {balanced_accuracy}\n")
        f.write(f"Confusion matrix: {conf_matrix}\n")
        f.write(f"Precision: {precision}\n")
        f.write(f"Recall: {recall}\n")
        f.write(f"F1: {f1}\n")
        f.write("\n")


# Dictionary of all models
models = {
    "PCA": {"table_name": "testPCAtransform", "input_dim": variables.PCA_components, "table_num": 4, "dropout": 0.4},
    "KPCA": {"table_name": "testKPCAtransform", "input_dim": variables.PCA_components, "table_num": 4, "dropout": 0.3},
    "PWdeepDAE": {"table_name": "testPWdeepDAEtransformed", "input_dim": variables.PCA_components, "table_num": 4, "dropout": 0.4},
    "deepDAE": {"table_name": "testdeepDAEtransformed", "input_dim": variables.PCA_components, "table_num": 4, "dropout": 0.0},
    "base": {"table_name": "test", "input_dim": variables.gene_number, "table_num": 46, "dropout": 0.3},
}


# SQL database
engine = create_engine(f"sqlite:///{variables.database_path}")


for data_type, data in models.items():

    test_table_name = data["table_name"]
    input_dim = data["input_dim"]
    table_num = data["table_num"]
    dropout = data["dropout"]

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

    test_ds = FE_Dataset(testX, testy)
    test_loader = DataLoader(test_ds, batch_size=512, num_workers=1)

    # Load the model
    model = Classifier(input_dim=input_dim, hidden_dim=variables.pathway_num, dropout_factor=dropout)
    model.load_state_dict(torch.load(f"{variables.classifier_model_path}/classifier_{data_type}.pt"))

    print(f"Model loaded: {data_type}")

    # Evaluate the model
    model.eval()

    # Predict
    evaluate(model, test_loader, data_type)

    print(f"Metrics saved for {data_type}")

    del model, test_ds, test_loader, testX, testy
    gc.collect()
