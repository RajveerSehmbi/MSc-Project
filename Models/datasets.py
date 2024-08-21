import sys
sys.path.append('/vol/bitbucket/rs218/MSc-Project')

from torch.utils.data import Dataset
import pandas as pd
from sqlalchemy import create_engine
import torch
import variables

# FOR USE IN FEATURE SELECTION CLASSIFIERS
class FS_Dataset(Dataset):

    def __init__(self, table, source, gene_num):

        self.cancer_types = variables.cancer_map

        engine = create_engine(f"sqlite:///{variables.database_path}")

        # Read the data from the table
        self.data = pd.read_sql(f"SELECT * FROM {table}_{source}", engine)

        top_genes = []
        with open(f"{variables.PCA_top_genes_file}", 'r') as f:
            for i, line in enumerate(f):
                # Gene is up to first colon
                gene = line.split(":")[0]
                top_genes.append(gene)
                if i == gene_num - 1:
                    break

        self.labels = self.data['cancer_type']
        self.data = self.data[top_genes]

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        
        X = self.data.iloc[idx]
        X = X.to_numpy()
        y = self.cancer_types[self.labels.iloc[idx]]

        # Convert to tensor
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.long)

        return X, y



# FOR USE IN FEATURE EXTRACTION CLASSIFIERS
class FE_Dataset(Dataset):

    def __init__(self, table):
        self.cancer_types = variables.cancer_map

        engine = create_engine(f"sqlite:///{variables.database_path}")

        if table in ['train', 'val', 'test', 'es']:
            self.table_num = 59
        elif table in ['trainPCAtransform', 'valPCAtransform', 'testPCAtransform', 'esPCAtransform']:
            self.table_num = 4

        # Read the data from the table
        self.data = pd.DataFrame()
        for i in range(0, self.table_num):
            table_name = f"{table}_{i}"
            df = pd.read_sql(f"SELECT * FROM {table_name}", engine, index_col='row_num')
            self.data = pd.concat([self.data, df], axis=1)
            print(f"Read {table}_{i}")

        self.labels = self.data['cancer_type']
        self.data = self.data.drop(columns=['cancer_type'])

        print("Dataset loaded with shape:", self.data.shape)


    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        X = self.data.iloc[idx]
        X = X.to_numpy()
        y = self.cancer_types[self.labels.iloc[idx]]

        # Convert to tensor
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.long)

        return X, y
