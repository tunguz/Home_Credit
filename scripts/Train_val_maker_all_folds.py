import numpy as np
import pandas as pd
from sklearn.model_selection StratifiedKFold


train = pd.read_csv('../input/train_1830.csv')
application_train = pd.read_csv('../input/application_train.csv')
target = application_train.TARGET.values

good_features = train.columns[2:]

kf = StratifiedKFold(5, shuffle=True, random_state=1974)

for i, (train_index, test_index) in enumerate(kf.split(train,target)):
    print(i)
    xgtrain, xgval = train[good_features].loc[train_index], train[good_features].loc[test_index]
    y_train, y_val = target[train_index], target[test_index]
        
    xgtrain['target'] = y_train
    xgval['target'] = y_val
        
    xgtrain.to_csv(f'../input/xgtrain_fold_{i}.csv', index=False)
    xgval.to_csv(f'../input/xgval_fold_{i}.csv', index=False)