import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import joblib


cancer_types = ["LAML", "ACC","CHOL","BLCA","BRCA","CESC","COAD","UCEC","ESCA",
                "GBM","HNSC","KICH","KIRC","KIRP","DLBC","LIHC","LGG",
                "LUAD","LUSC","SKCM","MESO","UVM","OV","PAAD","PCPG","PRAD",
                "READ","SARC","STAD","TGCT","THYM","THCA","UCS"]

#---------------- Combines all read count datasets into one and appends correct sample type label (Tumour or Normal) -----------------#
read_count_dfs = []

sample_types_map = {'Buccal Cell Normal':'Normal', 'Solid Tissue Normal':'Normal', 'Bone Marrow Normal':'Normal', 'Null':'Normal',
                'Primary Tumor':'Tumour', 'Primary Blood Derived Cancer - Peripheral Blood':'Tumour', 'Metastatic':'Tumour',
                'Recurrent Tumor':'Tumour', 'Additional - New Primary':'Tumour', 'Additional Metastatic':'Tumour',
                'FFPE Scrolls':'Tumour'}

for cancer in cancer_types:

    # Read Counts
    rcdf = pd.read_csv(f'Data/Read_Counts/TCGA-{cancer}.htseq_counts.tsv', delimiter='\t', index_col=0)
    rcdf = rcdf.transpose()
    ## remove columns that are not gene symbol
    rcdf = rcdf.drop(columns=['__no_feature', '__ambiguous', '__too_low_aQual', '__not_aligned', '__alignment_not_unique'])

    rcdf['cancer_type'] = cancer      ## sectioning between cancers for later splitting

    # Sample Types
    stdf = pd.read_csv(f'Data/Phenotype/{cancer}.tsv', delimiter='\t')
    stdf.set_index('sample', inplace=True)
    stdf = stdf.drop('samples',axis=1)
    stdf.rename(columns={'sample_type.samples': 'sample_type'}, inplace=True)

    # Attach correct label
    if (cancer == "LAML"):
        print(rcdf.index)
    rcdf = rcdf.join(stdf)
    rcdf['sample_type'] = rcdf['sample_type'].map(sample_types_map)

    read_count_dfs.append(rcdf)
    print(f'{cancer} completed.')

# Combine into big dataset
combined_data = pd.concat(read_count_dfs)
print("Combined all datasets.")



#------------------ Changing Ensembl IDs to Gene Symbols-------------------#
ensembl_to_gene_symbol = pd.read_csv('Data/Read_Counts/ensemblMapping.csv', delimiter='\t')

ensembl_to_gene_symbol.set_index('id', inplace=True)

combined_data.columns = combined_data.columns.map(lambda x: ensembl_to_gene_symbol.loc[x, 'gene'] if x in ensembl_to_gene_symbol.index else x)

print("Changed to gene symbols")


#------------------ generate table of sample sizes ------------------#
sample_counts = {}

for cancer in cancer_types:

    sample_counts[cancer] = combined_data.loc[combined_data['cancer_type'] == cancer, 'sample_type'].value_counts().to_dict()


sample_counts_df = pd.DataFrame.from_dict(sample_counts, orient='index').fillna(0)

sample_counts_df['Total'] = sample_counts_df['Tumour'] + sample_counts_df['Normal']

sample_counts_df.to_csv('Data/sample_counts_per_cancer.csv', sep='\t')

print("Finished count table")



#------------------ Creating training, validation and test sets ------------------#
combined_data.loc[combined_data['sample_type'] == 'Normal', 'cancer_type'] = 'Normal'

combined_data = combined_data.drop('sample_type', axis=1)

train_set = pd.DataFrame()
val_set = pd.DataFrame()
test_set = pd.DataFrame()

# 80-10-10 split
for cancer in combined_data['cancer_type'].unique():
    cancer_set = combined_data[combined_data['cancer_type'] == cancer]

    train, temp = train_test_split(cancer_set, test_size=0.2, random_state=2)
    val, test = train_test_split(temp, test_size=0.5, random_state=2)

    train_set = pd.concat([train_set, train])
    val_set = pd.concat([val_set, val])
    test_set = pd.concat([test_set, test])

print("finished making sets")

#------------------ z-score normalisation for training set ------------------#
train_y = train_set['cancer_type']
train_x = train_set.drop('cancer_type', axis=1)

scaler = StandardScaler()
zscore_normalised_train  = pd.DataFrame(scaler.fit_transform(train_x), index=train_x.index, columns=train_x.columns)

joblib.dump(scaler, 'Data/Scalers/z_score_normalisation_scaler.gz')  # save scaler for future use

print("Completed z-score normalisation")


#------------------ SMOTE balancing ------------------#

smote = SMOTE(random_state=2)
features_resampled, labels_resampled = smote.fit_resample(zscore_normalised_train, train_y)

print ("finished SMOTE pt1")

resampled_data = pd.concat([pd.DataFrame(features_resampled, columns=features.columns), 
                            pd.Series(labels_resampled, name='cancer_type')], axis=1)

print("finished SMOTE")




#------------------ save sets to csv files ------------------#
zscore_normalised_train.to_csv('train_set(zscore).csv', sep='\t')
val_set.to_csv('val_set.csv', sep='\t')
test_set.to_csv('test_set.csv', sep='\t')



