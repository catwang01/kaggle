from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from path import *
from utils import load_data
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


X_train, y_train, X_test, test_id, feature_names = load_data(reinbalancedDataPath)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, random_state=SEED, test_size=TEST_SIZE)

trainer = ortTrainer(modelClass=XGBClassifier ,
                     params=params,
                     tuned_params=tuned_params,
                     isupdate=True,
                     istune=True,
                     modelName='xgb')

trainer.fit( X_train, y_train,
    eval_set=[(X_val, y_val)],
    early_stopping_rounds=100,
    eval_metric="auc",
    verbose=True
)

trainer.getOutput()





