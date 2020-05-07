from xgboost import XGBClassifier
import numpy as np
from sklearn.externals import joblib
import pandas as pd
from utils import load_data, plotAuc, getNextVer
from path import *
from baseModel import ortTrainer



params = {
    'objective': 'binary:logistic',
    "booster": "gbtree",
    "learning_rate": 0.02,  # shrinkage
    "max_depth": 12,
    "subsample": 0.6,
    "colsample_bytree": 0.5,
    "seed": 1301,
    'n_jobs': -1
}

tuned_params = {
    "learning_rate": [0.02, 0.2, 0.5, 0.9],  # shrinkage
    "max_depth": [8, 12, 15, 20],
    "subsample": [0.6, 0.7, 0.8, 0.9],
    "colsample_bytree": [0.3, 0.5, 0.7, 0.9],
}

trainer = ortTrainer(modelClass=XGBClassifier,
                  params=params,
                  tuned_params=tuned_params,
                  isupdate=True,
                  istune=True,
                  modelName='xgb',
                  dataPath=processedDataPath)

trainer.read_data()
trainer.fit( eval_set=[(trainer.X_val, trainer.y_val)],
                       early_stopping_rounds=10,
                       eval_metric="auc",
                       verbose=True)
trainer.getOutput()

