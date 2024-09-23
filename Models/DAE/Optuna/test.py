import sys
sys.path.append('/vol/bitbucket/rs218/MSc-Project')

import pandas as pd
import variables

params = pd.read_csv(f"{variables.optuna_path}/PWdeepDAE_best_params.csv")

dropout_rate = params['params_dropout_rate'].iloc[0]

dropout_rate = int(dropout_rate)

print(dropout_rate)
