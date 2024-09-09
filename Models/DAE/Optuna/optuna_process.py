import sys
sys.path.append('/vol/bitbucket/rs218/MSc-Project')

import joblib
import optuna
import variables

study = joblib.load('deepDAE_optuna.pkl')
df = study.trials_dataframe().drop(['state','datetime_start','datetime_complete'], axis=1)

df.to_csv('deepDAE_optuna.csv', index=False)
print("Optuna study saved.")

# Save the top three trials in deepDAE_top3_params.csv, lowest loss first
df = df.sort_values('value', ascending=True)
df = df.head(3)
df.to_csv('deepDAE_top3_params.csv', index=False)
print("Top 3 params saved.")


# Visualize the study.
optuna.visualization.plot_optimization_history(study).write_image(f'{variables.image_path}/deepDAE_optuna_history.png')
optuna.visualization.plot_param_importances(study).write_image(f'{variables.image_path}/deepDAE_optuna_importance.png')
optuna.visualization.plot_slice(study).write_image(f'{variables.image_path}/deepDAE_optuna_slice.png')


study = joblib.load('PWdeepDAE_optuna.pkl')
df = study.trials_dataframe().drop(['state','datetime_start','datetime_complete'], axis=1)

df.to_csv('PWdeepSDAE_optuna.csv', index=False)
print("Optuna study saved.")

# Save the top three trials in deepDAE_top3_params.csv, lowest loss first
df = df.sort_values('value', ascending=True)
df = df.head(3)
df.to_csv('PWdeepSDAE_top3_params.csv', index=False)
print("Top 3 params saved.")

# Visualize the study.
optuna.visualization.plot_optimization_history(study).write_image(f'{variables.image_path}/PWdeepDAE_optuna_history.png')
optuna.visualization.plot_param_importances(study).write_image(f'{variables.image_path}/PWdeepDAE_optuna_importance.png')
optuna.visualization.plot_slice(study).write_image(f'{variables.image_path}/PWdeepDAE_optuna_slice.png')
