import sys
sys.path.append('/vol/bitbucket/rs218/MSc-Project')

import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score
from sqlalchemy import create_engine
import variables
import sys

engine = create_engine(f"sqlite:///{variables.database_path}")

def load_data(table_name):
        test = pd.DataFrame()
        for i in range(0, 4):
            table = pd.read_sql(f"SELECT * FROM {table_name}_{i}", engine, index_col='row_num')
            test = pd.concat([test, table], axis=1)
            print(f"Read {table_name}_{i}")
        return test

def main(name):


    # Load PCA and KPCA data
    print(f"Loading data for {name}...")
    test_data = load_data(f"test{name}")
    train_data = load_data(f"train{name}")
    print("Data loaded.")

    testX = test_data.drop(columns=['cancer_type'])
    testy = test_data['cancer_type']

    trainX = train_data.drop(columns=['cancer_type'])
    trainy = train_data['cancer_type']

    # Step 3: Use the existing cancer_map to map the string labels to numerical labels
    # Convert string labels to numerical labels using the map
    testy = testy.map(variables.cancer_map)

    # Step 4: Apply GMM clustering
    num_classes = len(variables.cancer_map)  # Use the length of the cancer map as the number of clusters
    gmm = GaussianMixture(n_components=num_classes, random_state=42, verbose=2, max_iter=500)  # GMM with the number of classes
    gmm.fit(trainX) 
    print("GMM training complete.")

    # Predict the cluster labels
    gmm_labels = gmm.predict(testX)
    print("GMM prediction complete.")


    # Step 5: Compute ARI between true labels and GMM cluster labels
    ari = adjusted_rand_score(testy, gmm_labels)
    print(f'Adjusted Rand Index (ARI): {ari}')

    # Save the ARI to a text file
    with open(f'gmm_ari_{name}.txt', 'w') as file:
        file.write(f'Adjusted Rand Index (ARI): {ari}')

if __name__ == '__main__':

    # Pass the name of the experiment as an argument
    name = sys.argv[1]
    print(f"{name} experiment started.")
    main(name)



