import sys
sys.path.append('/vol/bitbucket/rs218/MSc-Project')

import joblib
import optuna
import variables

if variables.DAE_type == 'standard':
    study = joblib.load('deepDAE_optuna.pkl')
    df = study.trials_dataframe().drop(['state','datetime_start','datetime_complete'], axis=1)

    df.to_csv('deepSDAE_optuna.csv', index=False)
    print("Optuna study saved.")

    # Visualize the study.
    optuna.visualization.plot_optimization_history(study).write_image(f'{variables.image_path}/deepSDAE_optuna_history.png')
    optuna.visualization.plot_param_importances(study).write_image(f'{variables.image_path}/deepSDAE_optuna_importance.png')
    optuna.visualization.plot_slice(study).write_image(f'{variables.image_path}/deepSDAE_optuna_slice.png')

elif variables.DAE_type == 'pathway':
    study = joblib.load('PWdeepDAE_optuna.pkl')
    df = study.trials_dataframe().drop(['state','datetime_start','datetime_complete'], axis=1)

    df.to_csv('PWdeepSDAE_optuna.csv', index=False)
    print("Optuna study saved.")

    # Visualize the study.
    optuna.visualization.plot_optimization_history(study).write_image(f'{variables.image_path}/PWdeepSDAE_optuna_history.png')
    optuna.visualization.plot_param_importances(study).write_image(f'{variables.image_path}/PWdeepSDAE_optuna_importance.png')
    optuna.visualization.plot_slice(study).write_image(f'{variables.image_path}/PWdeepSDAE_optuna_slice.png')
