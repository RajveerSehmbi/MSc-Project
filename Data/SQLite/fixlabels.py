import sys
sys.path.append('/vol/bitbucket/rs218/MSc-Project')

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sqlalchemy import create_engine
import variables

engine = create_engine(f"sqlite:///{variables.database_path}")


table_name = f"train_45"
train_labels = pd.read_sql(f"SELECT * FROM {table_name}", engine, index_col='row_num')
# Print the last column to check if it is the cancer type
print("Original cancer type.")
print(train_labels.iloc[:, -1])
print(train_labels.shape)

table_name_pca = f"trainPCAtransform_3"
train_labels_transform_pca = pd.read_sql(f"SELECT * FROM {table_name_pca}", engine, index_col='row_num')
# Print the last column to check if it is the cancer type
print("Original cancer type in PCA transformed data.")
print(train_labels_transform_pca.iloc[:, -1])
print(train_labels_transform_pca.shape)

table_name_kpca = f"trainKPCAtransform_3"
train_labels_transform_kpca = pd.read_sql(f"SELECT * FROM {table_name_kpca}", engine, index_col='row_num')
# Print the last column to check if it is the cancer type
print("Original cancer type in KPCA transformed data.")
print(train_labels_transform_kpca.iloc[:, -1])
print(train_labels_transform_kpca.shape)


# Change the cancer type in the PCA transformed data to match the original data
train_labels_transform_pca.iloc[:, -1] = train_labels.iloc[:, -1]
print("Changed cancer type in PCA transformed data.")
print(train_labels_transform_pca.iloc[:, -1])

# Change the cancer type in the KPCA transformed data to match the original data
train_labels_transform_kpca.iloc[:, -1] = train_labels.iloc[:, -1]
print("Changed cancer type in KPCA transformed data.")
print(train_labels_transform_kpca.iloc[:, -1])

# Add row num column to the PCA transformed data
train_labels_transform_pca['row_num'] = train_labels.index

# Add row num column to the KPCA transformed data
train_labels_transform_kpca['row_num'] = train_labels.index

# Save the corrected data to the SQLite database
train_labels_transform_pca.to_sql(table_name_pca, engine, if_exists='replace', index=False, index_label='row_num')
train_labels_transform_kpca.to_sql(table_name_kpca, engine, if_exists='replace', index=False, index_label='row_num')

print("Data corrected and saved.")

print("Checking data is correct for PCA.")
train_labels_transform_pca = pd.read_sql(f"SELECT * FROM {table_name_pca}", engine, index_col='row_num')

# For each row in the original data, check if the cancer type is the same in the PCA transformed data
for i in range(train_labels.shape[0]):
    if train_labels.iloc[i, -1] != train_labels_transform_pca.iloc[i, -1]:
        print(f"Error in row {i}.")
        print(f"Original: {train_labels.iloc[i, -1]}")
        print(f"Transformed: {train_labels_transform_pca.iloc[i, -1]}")

print("Data checked for PCA.")

print("Checking data is correct for KPCA.")
train_labels_transform_kpca = pd.read_sql(f"SELECT * FROM {table_name_kpca}", engine, index_col='row_num')

# For each row in the original data, check if the cancer type is the same in the KPCA transformed data
for i in range(train_labels.shape[0]):
    if train_labels.iloc[i, -1] != train_labels_transform_kpca.iloc[i, -1]:
        print(f"Error in row {i}.")
        print(f"Original: {train_labels.iloc[i, -1]}")
        print(f"Transformed: {train_labels_transform_kpca.iloc[i, -1]}")

print("Data checked for KPCA.")


table_name = f"test_45"
test_labels = pd.read_sql(f"SELECT * FROM {table_name}", engine, index_col='row_num')

table_name_pca = f"testPCAtransform_3"
test_labels_transform_pca = pd.read_sql(f"SELECT * FROM {table_name_pca}", engine, index_col='row_num')

table_name_kpca = f"testKPCAtransform_3"
test_labels_transform_kpca = pd.read_sql(f"SELECT * FROM {table_name_kpca}", engine, index_col='row_num')

# Check if the cancer type in the PCA transformed test data matches the original test data for each row
for i in range(test_labels.shape[0]):
    if test_labels.iloc[i, -1] != test_labels_transform_pca.iloc[i, -1]:
        print(f"Error in row {i} for PCA.")
        print(f"Original: {test_labels.iloc[i, -1]}")
        print(f"Transformed: {test_labels_transform_pca.iloc[i, -1]}")

print("Data checked for PCA.")

# Check if the cancer type in the KPCA transformed test data matches the original test data for each row
for i in range(test_labels.shape[0]):
    if test_labels.iloc[i, -1] != test_labels_transform_kpca.iloc[i, -1]:
        print(f"Error in row {i} for KPCA.")
        print(f"Original: {test_labels.iloc[i, -1]}")
        print(f"Transformed: {test_labels_transform_kpca.iloc[i, -1]}")

print("Data checked for KPCA.")