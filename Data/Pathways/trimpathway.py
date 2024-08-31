import sys
sys.path.append('/vol/bitbucket/rs218/MSc-Project')

from sqlalchemy import create_engine
import pandas as pd
import variables



engine = create_engine(f"sqlite:///{variables.database_path}")

# Read pathway_genes.csv
pathway_genes = pd.read_csv(variables.pathway_file)

# Read train data and keep columns
train_columns = pd.DataFrame()
for i in range(0, 46):
    table_name = f"train_{i}"
    df = pd.read_sql(f"SELECT * FROM {table_name}", engine, index_col='row_num')
    df = df.columns
    train_columns = pd.concat([train_columns, df], axis=0)
    print(f"Read train_{i}")

# Get the gene column from the pathway_genes.csv
all_genes = pathway_genes['gene']

# Get the genes that are not common between the train data and the pathway_genes.csv
gene_no_data = all_genes[~all_genes.isin(train_columns)]
print(gene_no_data)

# Remove the gene_no_data genes from the pathway_genes.csv
pathway_genes = pathway_genes[~pathway_genes['gene'].isin(gene_no_data)]

# Save the trimmed pathway_genes.csv
pathway_genes.to_csv(f"{variables.pathway_file}_trimmed", index=False)
