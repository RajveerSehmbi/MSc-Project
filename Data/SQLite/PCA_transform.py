import sys
sys.path.append('/vol/bitbucket/rs218/MSc-Project')

from sqlalchemy import create_engine
import pandas as pd
import joblib
import variables

engine = create_engine(f"sqlite:///{variables.database_path}")

models = {"PCA":joblib.load(variables.PCA_model_path), "KPCA": joblib.load(variables.kPCA_model_path)}

# Read training data
train = pd.DataFrame()
for i in range(0, variables.table_num):
    table_name = f"train_{i}"
    df = pd.read_sql(f"SELECT * FROM {table_name}", engine, index_col='row_num')
    train = pd.concat([train, df], axis=1)
    print(f"Read train_{i}")

train_labels = train['cancer_type']
train = train.drop(columns=['cancer_type'])
print("Training data loaded with shape:", train.shape)
print("Training labels loaded with shape:", train_labels.shape)

# Read test data
test = pd.DataFrame()
for i in range(0, variables.table_num):
    table_name = f"test_{i}"
    df = pd.read_sql(f"SELECT * FROM {table_name}", engine, index_col='row_num')
    test = pd.concat([test, df], axis=1)
    print(f"Read test_{i}")

test_labels = test['cancer_type']
test = test.drop(columns=['cancer_type'])
print("Test data loaded with shape:", test.shape)
print("Test labels loaded with shape:", test_labels.shape)


# Transform data
for table, model in models.items():

    print(f"Transforming data using {table} model...")
    train_transformed = model.transform(train)
    test_transformed = model.transform(test)
    print(f"Transformation complete for {table}.")

    # Reattach labels
    train_transformed = pd.DataFrame(train_transformed)
    train_transformed['cancer_type'] = train_labels
    test_transformed = pd.DataFrame(test_transformed)
    test_transformed['cancer_type'] = test_labels

    # Save transformed train data
    count = 0
    train_rows = train_transformed.shape[0]
    for i in range(0, train_transformed.shape[1], 999):
        table_name = f"train{table}transform_{count}"
        subset = train_transformed.iloc[:, i:i+999]
        subset['row_num'] = range(0, train_rows)
        subset.to_sql(table_name, engine, if_exists="replace", index=False, index_label='row_num')
        count += 1
        print(f"Loop {count} completed for {table}.")
    
    print(f"Train data saved for {table}.")

    # Save transformed test data
    count = 0
    test_rows = test_transformed.shape[0]
    for i in range(0, test_transformed.shape[1], 999):
        table_name = f"test{table}transform_{count}"
        subset = test_transformed.iloc[:, i:i+999]
        subset['row_num'] = range(0, test_rows)
        subset.to_sql(table_name, engine, if_exists="replace", index=False, index_label='row_num')
        count += 1
        print(f"Loop {count} completed for {table}.")

    print(f"Test data saved for {table}.")

print("Data transformation complete.")


