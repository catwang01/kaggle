from xgboost import XGBClassifier
import numpy as np
from sklearn.externals import joblib
import pandas as pd
from utils import load_data, plotAuc, getNextVer
from path import *
from baseModel import xgbTrainer



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
    "learning_rate": [0.02, 0.2, 0.5, 1],  # shrinkage
    "max_depth": [8, 12, 15],
    "subsample": [0.6, 0.7, 0.8],
    "colsample_bytree": [0.5, 0.7, 0.9],
}

trainer = xgbTrainer(modelClass=XGBClassifier,
                  other_params=params,
                  tuned_params=tuned_params,
                  isupdate=True,
                  istune=True,
                  modelName='xgb',
                  dataPath=processedDataPath)

trainer.read_data()
trainer.fit()
trainer.getOutput()

