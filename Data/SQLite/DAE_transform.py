import sys
sys.path.append('/vol/bitbucket/rs218/MSc-Project')


from Models.DAE.deepDAE import TrainedEncoder
import pandas as pd
from sqlalchemy import create_engine
import torch
import variables
import gc



def load_data(engine, table_name):

    data = pd.DataFrame()
    for i in range(0, variables.table_num):
        table = pd.read_sql(f"SELECT * FROM {table_name}_{i}", engine, index_col='row_num')
        data = pd.concat([data, table], axis=1)
        print(f"Read {table_name}_{i}")

    return data.drop(columns=['cancer_type']), data['cancer_type']


def save_data(engine, table_name, data):

    count = 0
    row_num = data.shape[0]
    for i in range(0, data.shape[1], 999):
        sql_table = f"{table_name}_{count}"
        subset = data.iloc[:, i:i+999].copy()
        subset['row_num'] = range(0, row_num)
        subset.to_sql(sql_table, engine, if_exists="replace", index=False, index_label='row_num')
        count += 1
        print(f"Loop {count} completed for {sql_table}")

    print(f"All data saved for {table_name}")


def transform(encoder, x, device):

    print(f"Original data shape: {x.shape}")

    # Save x index
    index = x.index

    # Convert dataframe to tensor
    x = torch.tensor(x.values).float().to(device)

    # Transform the data
    x = encoder(x)

    # Convert tensor to dataframe
    x = pd.DataFrame(x.cpu().detach().numpy(), index=index)

    print(f"Transformed data shape: {x.shape}")

    return x

def main():

    engine = create_engine(f"sqlite:///{variables.database_path}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_type = None
    params = None
    if variables.DAE_type == 'standard':
        params = pd.read_csv(f"{variables.optuna_path}/deepDAE_best_params.csv")
    elif variables.DAE_type == 'pathway':
        model_type = 'PWdeepDAE'
        params = pd.read_csv(f"{variables.optuna_path}/PWdeepDAE_best_params.csv")

    # Dropout_rate
    dropout_rate = params['params_dropout_rate'][0]
    dropout_rate = float(dropout_rate)

    # Load the encoder
    print("Loading encoder...")
    encoder = TrainedEncoder(dropout_rate).to(device)
    encoder.load_state_dict(torch.load(f"{variables.DAE_model_path}/DAE_encoder_{model_type}.pt"))
    encoder.eval()
    print("Encoder loaded.")

    # Load the train data
    print("Loading train data...")
    trainX, trainy = load_data(engine, 'train')
    print("Train data loaded.")

    # Transform the train data
    print("Transforming train data...")
    transformedX = transform(encoder, trainX, device)
    print("Train data transformed.")

    del trainX
    gc.collect()

    # Save the transformed train data
    print("Saving train data...")
    transformedX['cancer_type'] = trainy
    save_data(engine, f'train{model_type}transformed', transformedX)
    print("Train data saved.")

    del transformedX, trainy
    gc.collect()

    # Load the test data
    print("Loading test data...")
    testX, testy = load_data(engine, 'test')
    print("Test data loaded.")

    # Transform the test data
    print("Transforming test data...")
    transformedX = transform(encoder, testX, device)
    print("Test data transformed.")

    del testX
    gc.collect()

    # Save the transformed test data
    print("Saving test data...")
    transformedX['cancer_type'] = testy
    save_data(engine, f'test{model_type}transformed', transformedX)
    print("Test data saved.")






if __name__ == "__main__":
    main()