import sys
sys.path.append('/vol/bitbucket/rs218/MSc-Project')

from sqlalchemy import create_engine
import pandas as pd
import joblib
import variables
import gc

engine = create_engine(f"sqlite:///{variables.database_path}")

ipca = joblib.load("/Users/raj/MSc-Project/PCA/ipca_model_nosmote.pkl")

# Read training data 1000 rows at a time
train = pd.DataFrame()

for i in range(0, variables.table_num):
    table_name = f"train_{i}"
    df = pd.read_sql(f"SELECT * FROM {table_name}", engine, index_col='row_num')
    train = pd.concat([train, df], axis=1)
    print(f"Read train_{i}")
    print(train.shape)

print("Data read complete for train.")
print(train.shape)

train_labels = train['cancer_type']
train = train.drop(columns=['cancer_type'])

# Transform data for IPCA
train_transformed_ipca = ipca.transform(train)
train_transformed_ipca = pd.DataFrame(train_transformed_ipca)
print(f"Transformation complete for IPCA.")
print(train_transformed_ipca.shape)

# Keep first 3734 columns
train_transformed_ipca = train_transformed_ipca.iloc[:, :3734]

# Reattach labels
train_transformed_ipca['cancer_type'] = train_labels

# Save transformed train data IPCA
train_transformed_ipca.reset_index(drop=True, inplace=True)
count = 0
train_rows = train_transformed_ipca.shape[0]
for i in range(0, train_transformed_ipca.shape[1], 999):
    table_name = f"trainIPCAtransform_{count}"
    subset = train_transformed_ipca.iloc[:, i:i+999].copy()
    subset['row_num'] = range(0, train_rows)
    subset.to_sql(table_name, engine, if_exists="replace", index=False, index_label='row_num')
    count += 1
    print(f"Loop {count} completed for IPCA.")

print(f"Train transformed data saved for IPCA.")

print("Data transformation complete for train.")

del train_transformed_ipca
gc.collect()

# Read test data 1000 rows at a time
test = pd.DataFrame()

for i in range(0, variables.table_num):
    table_name = f"test_{i}"
    df = pd.read_sql(f"SELECT * FROM {table_name}", engine, index_col='row_num')
    test = pd.concat([test, df], axis=1)
    print(f"Read test_{i}")
    print(test.shape)

print("Data read complete for test.")

test_labels = test['cancer_type']
test = test.drop(columns=['cancer_type'])

# Transform data for IPCA
test_transformed_ipca = ipca.transform(test)
test_transformed_ipca = pd.DataFrame(test_transformed_ipca)
print(f"Transformation complete for IPCA.")

# Keep first 3734 columns
test_transformed_ipca = test_transformed_ipca.iloc[:, :3734]

# Reattach labels
test_transformed_ipca['cancer_type'] = test_labels

# Save transformed test data IPCA
test_transformed_ipca.reset_index(drop=True, inplace=True)
count = 0
test_rows = test_transformed_ipca.shape[0]
for i in range(0, test_transformed_ipca.shape[1], 999):
    table_name = f"testIPCAtransform_{count}"
    subset = test_transformed_ipca.iloc[:, i:i+999].copy()
    subset['row_num'] = range(0, test_rows)
    subset.to_sql(table_name, engine, if_exists="replace", index=False, index_label='row_num')
    count += 1
    print(f"Loop {count} completed for IPCA.")

print(f"Test transformed data saved for IPCA.")

print("Data transformation complete for test.")
