import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler

from sqlalchemy import create_engine

import joblib
import gc

engine = create_engine("sqlite:///../Data/data_scaled.db")
print("Connected to SQLite database.")

def save_to_sql(data, table_name):

    rows = data.shape[0]
    count = 0
    for i in range(0, data.shape[1], 999):
        table = f"{table_name}_{count}"
        subset = data.iloc[:, i:i+999]
        subset['row_num'] = range(0, rows)
        subset.to_sql(table, engine, if_exists="replace", index=False, index_label='row_num')
        count += 1
        print(f"Loop {count} completed.")
        print(f"{table} saved.")
    print(f"Finished {table_name}")



cancer_types = ["LAML", "ACC","CHOL","BLCA","BRCA","CESC","COAD","UCEC","ESCA",
                "GBM","HNSC","KICH","KIRC","KIRP","DLBC","LIHC","LGG",
                "LUAD","LUSC","SKCM","MESO","UVM","OV","PAAD","PCPG","PRAD",
                "READ","SARC","STAD","TGCT","THYM","THCA","UCS"]

#---------------- Combines train features andl labels into dataframe -----------------#

train_features = pd.read_csv('../Data/Normalised_Read_Counts/TRAINvst_features.csv', delimiter=',', index_col=0)
train_features = train_features.transpose()
print(train_features.shape)
print(train_features.index)
labels = pd.read_csv('../Data/Normalised_Read_Counts/TRAINvst_labels.csv', delimiter=',', index_col=0)
print(labels.shape)

# Replace dots with hyphens in the index
labels.index = labels.index.str.replace('.', '-')

train_data = pd.concat([train_features, labels], axis=1)

# Rename the last column to 'cancer_type'
train_data.rename(columns={train_data.columns[-1]: 'cancer_type'}, inplace=True)

del train_features, labels
gc.collect()

print(train_data.shape)
print("Combined train.")

# Last 5 columns
print(train_data.columns[-5:])


#---------------- Combines test features andl labels into dataframe -----------------#

test_features = pd.read_csv('../Data/Normalised_Read_Counts/TESTvst_features.csv', delimiter=',', index_col=0)
test_features = test_features.transpose()
print(test_features.shape)
print(test_features.index)
labels = pd.read_csv('../Data/Normalised_Read_Counts/TESTvst_labels.csv', delimiter=',', index_col=0)
print(labels.shape)

# Replace dots with hyphens in the index
labels.index = labels.index.str.replace('.', '-')

test_data = pd.concat([test_features, labels], axis=1)

# Rename the last column to 'cancer_type'
test_data.rename(columns={test_data.columns[-1]: 'cancer_type'}, inplace=True)

del test_features, labels
gc.collect()

print(test_data.shape)
print("Combined train.")

# Last 5 columns
print(test_data.columns[-5:])


# #------------------ Changing Ensembl IDs to Gene Symbols-------------------#
# ensembl_to_gene_symbol = pd.read_csv('Data/Read_Counts/ensemblMapping.csv', delimiter='\t')

# ensembl_to_gene_symbol.set_index('id', inplace=True)

# combined_data.columns = combined_data.columns.map(lambda x: ensembl_to_gene_symbol.loc[x, 'gene'] if x in ensembl_to_gene_symbol.index else x)

# print("Changed to gene symbols")


#------------------ generate table of sample sizes ------------------#


def create_sample_counts_table(data, file_name):

    sample_counts = data['cancer_type'].value_counts().to_dict()
    sample_counts_df = pd.DataFrame.from_dict(sample_counts, orient='index').fillna(0)
    sample_counts_df.to_csv(f'../Data/Set_Counts/{file_name}', sep='\t')
    print(f"Finished count table: '{file_name}'")



create_sample_counts_table(train_data, 'sample_counts_train_scaled.tsv')
create_sample_counts_table(test_data, 'sample_counts_test_scaled.tsv')


# ------------------ Standard scaling ------------------ #

train_y = train_data['cancer_type']
train_x = train_data.drop('cancer_type', axis=1)

scaler = StandardScaler()
scaled_train_x  = pd.DataFrame(scaler.fit_transform(train_x), index=train_x.index, columns=train_x.columns)

del train_x
gc.collect()

scaled_train = pd.concat([scaled_train_x, train_y], axis=1)
print("Completed scaling for training set")

del train_y
gc.collect()


test_y = test_data['cancer_type']
test_x = test_data.drop('cancer_type', axis=1)
scaled_test_x = pd.DataFrame(scaler.transform(test_x), index=test_x.index, columns=test_x.columns)
scaled_test = pd.concat([scaled_test_x, test_y], axis=1)
print("Completed scaling for test set")

joblib.dump(scaler, '../Data/Scalers/standard_scaler.gz')

del test_x, test_y
gc.collect()

save_to_sql(scaled_test, 'test')
print("Saved test_set")

del scaled_test
gc.collect()


#------------------ SMOTE balancing for training set ------------------#

train_y = scaled_train['cancer_type']
train_x = scaled_train.drop('cancer_type', axis=1)

smote = SMOTE(random_state=2)
train_x_resampled, train_y_resampled = smote.fit_resample(train_x, train_y)

print ("finished first part SMOTE")
del train_y, smote, scaled_train
gc.collect()

train_x_resampled_df = pd.DataFrame(train_x_resampled, columns=train_x.columns, dtype='float64')

del train_x, train_x_resampled
gc.collect()

train_y_resampled_df = pd.Series(train_y_resampled, name='cancer_type').astype(str).astype('category')

if train_x_resampled_df.index.duplicated().any():
    train_x_resampled_df.reset_index(drop=True, inplace=True)
    train_y_resampled_df.reset_index(drop=True, inplace=True)

print(train_x_resampled_df.shape)
print(train_y_resampled_df.shape)

train_resampled_data = pd.concat([train_x_resampled_df, train_y_resampled_df], axis=1, copy=False).reset_index(drop=True)

print(train_resampled_data.shape)

print("finished final part SMOTE")
del train_y_resampled, train_x_resampled_df, train_y_resampled_df
gc.collect()

create_sample_counts_table(train_resampled_data, 'sample_counts_train_resampled_scaled.tsv')

print("Sample counts table created for train_resampled_data")


#------------------ save sets to SQL ------------------#

save_to_sql(train_resampled_data, 'train')
print("Saved train_set")






