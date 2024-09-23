import sys
sys.path.append('/Users/raj/MSc-Project')

import matplotlib.pyplot as plt
import pandas as pd
from sqlalchemy import create_engine
import seaborn as sns

engine = create_engine(f"sqlite:///../Data/SQLite/data_scaled2.db")

# # Load in PCA data from SQLite, keeping first 2 columns
# table_name = f"trainIPCAtransform_0"
# train_2d = pd.read_sql(f"SELECT * FROM {table_name}", engine, index_col='row_num')
# # Keep first 2 columns
# train_2d = train_2d.iloc[:, :2]
# print(train_2d.shape)
# print(f"Read train_0.")

# table_name = f"trainIPCAtransform_2"
# train_labels = pd.read_sql(f"SELECT * FROM {table_name}", engine, index_col='row_num')
# # Print the last column to check if it is the cancer type
# print(train_labels.iloc[:, -1])
# train_labels = train_labels['cancer_type']

# print(train_labels)
# # Print the unique values in the cancer type column
# print(train_labels.unique())

# print(f"Read train_2.")

# train_2d['cancer_type'] = train_labels
# print(train_2d.head())

# # Plot the data
# plt.close()
# plt.figure(figsize=(10, 10))
# sns.scatterplot(x=train_2d.columns[0], y=train_2d.columns[1], hue='cancer_type', data=train_2d, palette='hls', s=10)
# plt.title("IPCA: PC1 vs PC2 for Training Data")
# plt.xlabel("PC1")
# plt.ylabel("PC2")
# # Save the plot
# plt.show()
# plt.savefig('IPCA_2d_cluster.png')
# plt.close()




# Load in PCA data from SQLite, keeping first 2 columns
table_name = f"trainPCAtransform_0"
train_2d = pd.read_sql(f"SELECT * FROM {table_name}", engine, index_col='row_num')
# Keep first 2 columns
train_2d = train_2d.iloc[:, :2]
print(train_2d.shape)
print(f"Read train_0.")

table_name = f"trainPCAtransform_3"
train_labels = pd.read_sql(f"SELECT * FROM {table_name}", engine, index_col='row_num')
# Print the last column to check if it is the cancer type
print(train_labels.iloc[:, -1])
train_labels = train_labels['cancer_type']

print(train_labels)
# Print the unique values in the cancer type column
print(train_labels.unique())

print(f"Read train_3.")

train_2d['cancer_type'] = train_labels
print(train_2d.head())

# Plot the data in sns plot
plt.figure(figsize=(10, 10))
sns.scatterplot(x=train_2d.columns[0], y=train_2d.columns[1], hue='cancer_type', data=train_2d, palette='hls', s=10)
plt.title("PCA: PC1 vs PC2 for Training Data")
plt.xlabel("PC1")
plt.ylabel("PC2")
# Save the plot
plt.savefig('PCA_2d_cluster.png')
plt.close()


# Load in KPCA data from SQLite, keeping first 2 columns
table_name = f"trainKPCAtransform_0"
train_2d = pd.read_sql(f"SELECT * FROM {table_name}", engine, index_col='row_num')
# Keep first 2 columns
train_2d = train_2d.iloc[:, :2]
print(train_2d.shape)
print(f"Read train_0.")

table_name = f"trainKPCAtransform_3"
train_labels = pd.read_sql(f"SELECT * FROM {table_name}", engine, index_col='row_num')
train_labels = train_labels['cancer_type']
print(f"Read train_3.")

train_2d['cancer_type'] = train_labels
print(train_2d.head())

# Plot the data in sns plot
plt.figure(figsize=(10, 10))
sns.scatterplot(x=train_2d.columns[0], y=train_2d.columns[1], hue='cancer_type', data=train_2d, palette='hls', s=10)
plt.title("PCA: PC1 vs PC2 for Training Data")
plt.xlabel("PC1")
plt.ylabel("PC2")
# Save the plot
plt.savefig('KPCA_2d_cluster.png')
plt.close()


# Close the SQLite connection

engine.dispose()
print("Connection closed.")

