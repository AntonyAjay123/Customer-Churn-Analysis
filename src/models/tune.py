import optuna
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import recall_score
from sklearn.model_selection import cross_val_score

#Objective function for Optuna
def tune_model(X, y):
    def objective(trial):
        params={
            "n_estimators":trial.suggest_int("n_estimators",200,800),
            "learning_rate":trial.suggest_float("learning_rate",0.01,0.1),
            "num_leaves":trial.suggest_int("num_leaves",10,100),
            "max_depth":trial.suggest_int("max_depth",3,10),
            "min_child_samples":trial.suggest_int("min_child_samples",10,100),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0, 5),
            "reg_lambda": trial.suggest_float("reg_lambda", 0, 5),
            "scale_pos_weight":(y==0).sum()/(y==1).sum(),
            "random_state":42,
            "n_jobs":-1
            }
        model = LGBMClassifier(**params)
        scores = cross_val_score(model,X,y,cv=3,scoring='recall')
        return scores.mean()
    #Run Optuna
    study = optuna.create_study(direction="maximize")
    study.optimize(objective,n_trials=20)
    print("Best Params:", study.best_params)
    return study.best_params