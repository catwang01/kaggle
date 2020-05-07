import numpy as np
from sklearn.externals import joblib
import pandas as pd
from utils import load_data, plotAuc, getNextVer
from path import *
from baseModel import Trainer
from lightgbm import LGBMClassifier

class lgbTrainer(Trainer):

    def _fit(self):
        self.model = self.modelClass(**self.other_params)
        print("==================================================")
        print("Train Model {}".format(self.modelName))
        self.model.fit(self.X_train, self.y_train,
                  eval_set=[(self.X_val, self.y_val)],
                  early_stopping_rounds=10,
                  eval_metric="auc",
                  verbose=True)

        y_train_hat = self.model.predict_proba(self.X_train)[:, 1]
        y_val_hat = self.model.predict_proba(self.X_val)[:, 1]

        self.trainAuc = plotAuc(self.y_train, y_train_hat, "train")
        self.valAuc = plotAuc(self.y_val, y_val_hat, "val")

        print("Train auc: {} Val auc: {}".format(self.trainAuc, self.valAuc))
        joblib.dump(self.model, "{}".format(self.modelName + '.model'))
        print("Save Model {}".format(self.modelName))
        return self.model


params = {
    "boosting_type": "gbdt",
    "learning_rate": 0.1,  # shrinkage
    # "max_depth": 8,
    "num_leaves": 2**9,
    "n_estimators": 1000,
    "subsample": 0.7,
    "colsample_bytree": 1,
    "seed": 1301,
    'n_jobs': -1,
    'reg_lambda': 0.2,
    'silent': 1,
}

tuned_params = {
    "learning_rate": [0.02, 0.2, 0.5, 1],  # shrinkage
    # "max_depth": [8, 12, 15],
    "subsample": [0.6, 0.7, 0.8],
    "colsample_bytree": [0.5, 0.7, 0.9],
}

trainer = lgbTrainer(modelClass=LGBMClassifier,
                  other_params=params,
                  tuned_params=tuned_params,
                  isupdate=True,
                  istune=False,
                  modelName='lgb',
                  dataPath=processedDataPath)

trainer.read_data()
trainer.fit()
trainer.getOutput()

