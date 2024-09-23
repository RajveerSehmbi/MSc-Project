import sys
sys.path.append('/Users/raj/MSc-Project')

from sqlalchemy import create_engine
import pandas as pd
import joblib
import variables
import gc

engine = create_engine(f"sqlite:///{variables.database_path}")

models = {"PCA":joblib.load(variables.PCA_model_path), "KPCA": joblib.load(variables.kPCA_model_path)}

# Read training data 1000 rows at a time
train_transformed_pca = pd.DataFrame()
train_transformed_kpca = pd.DataFrame()


for i in range (0, 31000, 10000):
    train_1000 = pd.DataFrame()
    for j in range(0, variables.table_num):
        table_name = f"train_{j}"
        df = pd.read_sql(f"SELECT * FROM {table_name}", engine, index_col='row_num')
        # Keep only 10000 rows and all columns
        df = df.iloc[i:i+10000, :]
        print(df.shape)
        train_1000 = pd.concat([train_1000, df], axis=1)
        print(f"Read train_{j} with rows {i} to {i+10000}")
    
    # Remove last column
    train_labels = train_1000['cancer_type']
    train_1000 = train_1000.drop(columns=['cancer_type'])

    # Transform data for PCA
    train_transformed_pca_1000 = models['PCA'].transform(train_1000)
    train_transformed_pca_1000 = pd.DataFrame(train_transformed_pca_1000)
    print(f"Transformation complete for PCA with rows {i} to {i+10000}.")

    # Keep first 3734 columns
    train_transformed_pca_1000 = train_transformed_pca_1000.iloc[:, :3734]

    # Reattach labels and add to all data
    train_transformed_pca_1000['cancer_type'] = train_labels
    train_transformed_pca = pd.concat([train_transformed_pca, train_transformed_pca_1000], axis=0)
    print(f"Train data added to all data for PCA with rows {i} to {i+10000}.")

    del train_transformed_pca_1000
    gc.collect()

    # Transform data for KPCA
    train_transformed_kpca_1000 = models['KPCA'].transform(train_1000)
    train_transformed_kpca_1000 = pd.DataFrame(train_transformed_kpca_1000)
    print(f"Transformation complete for KPCA with rows {i} to {i+10000}.")

    # Keep first 3734 columns
    train_transformed_kpca_1000 = train_transformed_kpca_1000.iloc[:, :3734]

    # Reattach labels and add to all data
    train_transformed_kpca_1000['cancer_type'] = train_labels
    train_transformed_kpca = pd.concat([train_transformed_kpca, train_transformed_kpca_1000], axis=0)
    print(f"Train data added to all data for KPCA with rows {i} to {i+10000}.")

# Save transformed train data PCA
train_transformed_pca.reset_index(drop=True, inplace=True)
count = 0
train_rows = train_transformed_pca.shape[0]
for i in range(0, train_transformed_pca.shape[1], 999):
    table_name = f"trainPCAtransform_{count}"
    subset = train_transformed_pca.iloc[:, i:i+999].copy()
    subset['row_num'] = range(0, train_rows)
    subset.to_sql(table_name, engine, if_exists="replace", index=False, index_label='row_num')
    count += 1
    print(f"Loop {count} completed for PCA.")

print(f"Train transformed data saved for PCA.")

# Save transformed train data KPCA
train_transformed_kpca.reset_index(drop=True, inplace=True)
count = 0
train_rows = train_transformed_kpca.shape[0]
for i in range(0, train_transformed_kpca.shape[1], 999):
    table_name = f"trainKPCAtransform_{count}"
    subset = train_transformed_kpca.iloc[:, i:i+999].copy()
    subset['row_num'] = range(0, train_rows)
    subset.to_sql(table_name, engine, if_exists="replace", index=False, index_label='row_num')
    count += 1
    print(f"Loop {count} completed for KPCA.")

print(f"Train transformed data saved for KPCA.")

print("Data transformation complete for train.")

del train_transformed_pca, train_transformed_kpca
gc.collect()



# Read test data 1000 rows at a time
test = pd.DataFrame()

for j in range(0, variables.table_num):
    table_name = f"test_{j}"
    df = pd.read_sql(f"SELECT * FROM {table_name}", engine, index_col='row_num')
    print(df.shape)
    test = pd.concat([test, df], axis=1)
    print(f"Read test_{j}")

# Remove last column
test_labels = test['cancer_type']
test = test.drop(columns=['cancer_type'])

# Transform data for PCA
test_transformed_pca = models['PCA'].transform(test)
test_transformed_pca = pd.DataFrame(test_transformed_pca)
print(f"Transformation complete for PCA.")

# Keep first 3734 columns
test_transformed_pca = test_transformed_pca.iloc[:, :3734]

# Reattach labels
test_transformed_pca['cancer_type'] = test_labels

# Save transformed test data PCA
count = 0
test_rows = test_transformed_pca.shape[0]
for i in range(0, test_transformed_pca.shape[1], 999):
    table_name = f"testPCAtransform_{count}"
    subset = test_transformed_pca.iloc[:, i:i+999].copy()
    subset['row_num'] = range(0, test_rows)
    subset.to_sql(table_name, engine, if_exists="replace", index=False, index_label='row_num')
    count += 1
    print(f"Loop {count} completed for PCA.")

print(f"Test transformed data saved for PCA.")

del test_transformed_pca
gc.collect()

# Transform data for KPCA
test_transformed_kpca = models['KPCA'].transform(test)
test_transformed_kpca = pd.DataFrame(test_transformed_kpca)
print(f"Transformation complete for KPCA.")

# Keep first 3734 columns
test_transformed_kpca = test_transformed_kpca.iloc[:, :3734]

# Reattach labels
test_transformed_kpca['cancer_type'] = test_labels

# Save transformed test data KPCA
test_transformed_kpca.reset_index(drop=True, inplace=True)
count = 0
test_rows = test_transformed_kpca.shape[0]
for i in range(0, test_transformed_kpca.shape[1], 999):
    table_name = f"testKPCAtransform_{count}"
    subset = test_transformed_kpca.iloc[:, i:i+999].copy()
    subset['row_num'] = range(0, test_rows)
    subset.to_sql(table_name, engine, if_exists="replace", index=False, index_label='row_num')
    count += 1
    print(f"Loop {count} completed for KPCA.")

print(f"Test transformed data saved for KPCA.")

print("Data transformation complete for test.")


