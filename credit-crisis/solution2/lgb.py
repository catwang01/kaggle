import numpy as np
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
import pandas as pd
from utils import load_data
from path import *
from baseModel import ortTrainer
from lightgbm import LGBMClassifier

params = {
    "boosting_type": "gbdt",
    "learning_rate": 0.02,  # shrinkage
    "num_leaves": 2**11,
    "subsample": 0.8,
    "colsample_bytree": 0.7,
    "seed": SEED,
    'n_jobs': -1,
    'reg_lambda': 0.8,
    'max_depth': 8,
    'silent': 0,
    "n_estimators": 500,
}

tuned_params = {
    "learning_rate": [0.02, 0.2, 0.5, 0.8],  # shrinkage
    "max_depth": [8, 12, 15, 20],
    "subsample": [0.5, 0.6, 0.7, 0.8],
    'reg_lambda': [0.01, 0.2, 0.5, 0.8],
    "colsample_bytree": [0.3, 0.5, 0.7, 0.8],
}

# 训练模型
X_train, y_train, X_test, test_id, feature_names = load_data(reinbalancedDataPath)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, random_state=SEED, test_size=TEST_SIZE)
trainer = ortTrainer(modelClass=LGBMClassifier,
                     params=params,
                     tuned_params=tuned_params,
                     isupdate=True,
                     istune=False,
                     modelName='lgb')

trainer.fit( X_train, y_train,
    eval_set=[(X_val, y_val)],
    early_stopping_rounds=100,
    eval_metric="auc",
    verbose=True
)

trainer.getOutput(X_test, test_id, X_val, y_val)

print(trainer.bestParams)

dict(zip(trainer.model.feature_importances_, feature_names))


# # 查看所有的结果
# print(trainer.ort_res)
#
# 查看极差分析结果（可以根据正交实验的结果和训练时间，手动调整超参数）


