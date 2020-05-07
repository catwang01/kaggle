import numpy as np
from sklearn.externals import joblib
import pandas as pd
from utils import load_data, plotAuc, getNextVer
from path import *
from my_ort import ORT
from baseModel import ortTrainer
from lightgbm import LGBMClassifier


params = {
    "boosting_type": "gbdt",
    "learning_rate": 0.1,  # shrinkage
    # "max_depth": 8,
    "num_leaves": 2**9,
    # "n_estimators": 1000,
    "subsample": 0.7,
    "colsample_bytree": 1,
    "seed": SEED,
    'n_jobs': -1,
    'silent': 0,
}

tuned_params = {
    "learning_rate": [0.02, 0.2, 0.5, 0.8],  # shrinkage
    "max_depth": [8, 12, 15, 20],
    "subsample": [0.5, 0.6, 0.7, 0.8],
    'reg_lambda': [0.01, 0.2, 0.5, 0.8],
    "colsample_bytree": [0.3, 0.5, 0.7, 0.8],
    # "seed": [1, 2, 3, 4]
}


# 训练模型
trainer = ortTrainer(modelClass=LGBMClassifier,
                     params=params,
                     tuned_params=tuned_params,
                     isupdate=True,
                     istune=True,
                     modelName='lgb',
                     dataPath=reinbalancedDataPath)


trainer.read_data()
trainer.fit(
    eval_set=[(trainer.X_val, trainer.y_val)],
    early_stopping_rounds=10,
    eval_metric="auc",
    verbose=True
)

trainer.getOutput()

print(trainer.bestParams)

# # 查看所有的结果
# print(trainer.ort_res)
#
# 查看极差分析结果（可以根据正交实验的结果和训练时间，手动调整超参数）


