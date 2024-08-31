import sys
sys.path.append('/vol/bitbucket/rs218/MSc-Project')

from sqlalchemy import create_engine
import pandas as pd
import variables



engine = create_engine(f"sqlite:///{variables.database_path}")

# Read pathway_genes.csv
pathway_genes = pd.read_csv(variables.pathway_file)
print(pathway_genes.shape)

# Read train data and keep columns
train_columns = []
for i in range(0, 46):
    table_name = f"train_{i}"
    df = pd.read_sql(f"SELECT * FROM {table_name}", engine, index_col='row_num')
    # Keep only the column names
    df = df.columns
    # Save list of ciolumn names
    train_columns.extend(df)
    print(len(train_columns))
    print(f"Read train_{i}")

# Keep everything up to the first dot in the column names
train_columns = [x.split('.')[0] for x in train_columns]

# Get the gene column from the pathway_genes.csv
all_genes = pathway_genes['gene']
print(all_genes.shape)

# Get the genes that are not common between the train data and the pathway_genes.csv
gene_no_data = all_genes[~all_genes.isin(train_columns)]
print(gene_no_data)

# Remove the gene_no_data genes from the pathway_genes.csv
pathway_genes = pathway_genes[~pathway_genes['gene'].isin(gene_no_data)]
print(pathway_genes.shape)

# Save the trimmed pathway_genes.csv
pathway_genes.to_csv(f"{variables.pathway_file}_trimmed", index=False)
