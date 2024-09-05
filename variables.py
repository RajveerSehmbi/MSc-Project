# GENERAL
path = '/vol/bitbucket/rs218/MSc-Project'
#path = '/Users/raj/MSc-Project'

# DATA
database_path = f'{path}/Data/SQLite/data_scaled2.db'

cancer_map = {"LAML": 0, "ACC": 1, "CHOL": 2, "BLCA": 3, "BRCA": 4, "CESC": 5, "COAD": 6, "UCEC": 7, "ESCA": 8,
                "GBM": 9, "HNSC": 10, "KICH": 11, "KIRC": 12, "KIRP": 13, "DLBC": 14, "LIHC": 15, "LGG": 16,
                "LUAD": 17, "LUSC": 18, "SKCM": 19, "MESO": 20, "UVM": 21, "OV": 22, "PAAD": 23, "PCPG": 24, "PRAD": 25,
                "READ": 26, "SARC": 27, "STAD": 28, "TGCT": 29, "THYM": 30, "THCA": 31, "UCS": 32, "Normal": 33}
gene_number = 45684
pathway_file = f'{path}/Data/Pathways/pathway_genes.csv'
gene_order_file = f'{path}/Data/input_order.csv'
pathway_num = 127
table_num = 46


# EARLY STOPPING
es_path = f'{path}/temp'

# PCA
PCA_model_path = f'{path}/PCA/pca_model_scaled.pkl'
kPCA_model_path = f'{path}/PCA/kpca_model_scaled.pkl'
PCA_components = 3734
PCA_top_genes_file = f'{path}/PCA/top_512_genes.txt'


# DAE
DAE_type = 'standard'
optuna_path = f'{path}/Models/DAE/Optuna'
image_path = f'{path}/Models/DAE/Plots'
DAE_model_path = f'{path}/Models/DAE/Trained'

# CLASSIFIER
classifier_model_path = f'{path}/Models/Classifier/Trained'
optuna_path_classifier = f'{path}/Models/Classifier/Optuna'