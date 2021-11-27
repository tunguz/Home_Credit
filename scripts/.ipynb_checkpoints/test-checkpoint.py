import os
from dask.distributed import Client
from dask_cuda import LocalCUDACluster
from dask import dataframe as dd
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import numpy as np
import optuna
import gc

num_round = 1000

def objective(client, dtrain, dtest, test_y, trial):
    params = {
        'objective': trial.suggest_categorical('objective',['binary:logistic']),
        'tree_method': trial.suggest_categorical('tree_method',['gpu_hist']),  # 'gpu_hist','hist'
        'lambda': trial.suggest_loguniform('lambda',1e-3,10.0),
        'alpha': trial.suggest_loguniform('alpha',1e-3,10.0),
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.3,1.0),
        'subsample': trial.suggest_uniform('subsample', 0.4, 1.0),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.001,0.1),
        #'n_estimators': trial.suggest_categorical('n_estimators', [1000]),
        'max_depth': trial.suggest_categorical('max_depth', [3,5,7,9,11,13,15,17,20]),
        #'random_state': trial.suggest_categorical('random_state', [24,48,2020]),
        'min_child_weight': trial.suggest_int('min_child_weight', 1,300),
        'eval_metric': trial.suggest_categorical('eval_metric',['logloss']),
    }

    output = xgb.dask.train(client, params, dtrain, num_round)

    booster = output['booster']  # booster is the trained model
    booster.set_param({'predictor': 'gpu_predictor'})

    predictions = xgb.dask.predict(client, booster, dtest)
    predictions = predictions.compute()

    roc = roc_auc_score(test_y, predictions)

    return roc

def main():
    train_x = dd.read_csv('/home/data/xgtrain.csv')
    test_x = dd.read_csv('/home/data/xgval.csv')

    train_x = train_x.replace([np.inf, -np.inf], np.nan)
    train_y = train_x['target']
    train_x = train_x[train_x.columns.difference(['target'])]

    test_x = test_x.replace([np.inf, -np.inf], np.nan)
    test_y = test_x['target']
    test_x = test_x[test_x.columns.difference(['target'])]

    with LocalCUDACluster(n_workers=4) as cluster:
        client = Client(cluster)
        dtrain = xgb.dask.DaskDMatrix(client, train_x, train_y)
        dtest = xgb.dask.DaskDMatrix(client, test_x, test_y)

        study = optuna.create_study(direction='maximize')
        study.optimize(lambda trial: objective(client, dtrain, dtest, test_y, trial), n_trials=50)

if __name__ == "__main__":
    main()
