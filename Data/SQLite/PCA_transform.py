import sys
sys.path.append('/Users/raj/MSc-Project/Project')

from sqlalchemy import create_engine
import pandas as pd
import joblib
import variables

engine = create_engine(f"sqlite:///{variables.database_path}")

tables = ['train', 'es', 'test', 'val']

# Read the data from the tables
for table in tables:

    data = pd.DataFrame()
    for i in range(0,59):
        table_name = f"{table}_{i}"
        df = pd.read_sql(f"SELECT * FROM {table_name}", engine, index_col='row_num')
        data = pd.concat([data, df], axis=1)
        print(f"Read {table}_{i}")

    labels = data['cancer_type']
    data = data.drop(columns=['cancer_type'])

    print("Dataset loaded with shape:", data.shape)

    ipca = joblib.load(f'{variables.PCA_model_path}')
    data = ipca.transform(data)

    # Keep the top 95% explained variance
    data = data[:, 0:variables.PCA_components]

    # Add the labels back to the data
    data = pd.DataFrame(data)
    data['cancer_type'] = labels

    print(data.shape)

    count = 0
    train_rows = data.shape[0]
    for i in range(0, data.shape[1], 999):
        table_name = f"{table}PCAtransform_{count}"
        subset = data.iloc[:, i:i+999]
        subset['row_num'] = range(0, train_rows)
        subset.to_sql(table_name, engine, if_exists="replace", index=False, index_label='row_num')
        count += 1
        print(f"Loop {count} completed.")


