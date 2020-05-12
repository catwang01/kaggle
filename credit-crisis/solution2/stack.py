import numpy as np
from sklearn.model_selection import KFold, train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier
from lightgbm import LGBMClassifier
from utils import load_data
from baseModel import ortTrainer, baseTrainer
from path import *

NFOLDS = 3
kf = KFold(n_splits=NFOLDS, random_state=SEED)

X_train, y_train, X_test, test_id, feature_names = load_data(reinbalancedDataPath)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, random_state=SEED, test_size=TEST_SIZE)

class TrainerWrapper:
    def __init__(self, modelClass, params):
        self.model = modelClass(**params)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def feature_importances(self, X, y):
        return self.model.fit(X, y).feature_importances_


class LgbWrapper(TrainerWrapper):
    def fit(self, X, y):
        X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=SEED, test_size=TEST_SIZE)
        self.model.fit(X_train, y_train,
                       eval_set=[(X_val, y_val)],
                       early_stopping_rounds=10, verbose=True)

class XgBoostWrapper(TrainerWrapper):
    def fit(self, X, y):
        X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=SEED, test_size=TEST_SIZE)
        self.model.fit(X_train, y_train,
                       eval_set=[(X_val, y_val)],
                       early_stopping_rounds=10,
                       eval_metric="auc", verbose=False)


def get_oof(clf, x_train, y_train, x_test):
    ntrain = x_train.shape[0]
    ntest = x_test.shape[0]
    oof_train = np.zeros((ntrain,))
    oof_test = np.zeros((ntest,))
    oof_test_skf = np.empty((NFOLDS, ntest))  #NFOLDS行，ntest列的二维array
    for i, (train_index, test_index) in enumerate(kf.split(x_train)): #循环NFOLDS次
        x_tr = x_train[train_index]
        y_tr = y_train[train_index]
        x_te = x_train[test_index]

        clf.fit(x_tr, y_tr)
        oof_train[test_index] = clf.predict(x_te)
        oof_test_skf[i, :] = clf.predict(x_test)

    oof_test[:] = oof_test_skf.mean(axis=0)
    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)

rf_params = {
    'n_jobs': -1,
    'n_estimators': 20,
    'max_depth': 12,
    'random_state': SEED
}

rf_tune_params = {
    'n_estimators': [10, 20, 30, 40, 50, 60],
    'max_depth': [8, 9, 10, 12, 15, 20],
}

rftrainer_params = dict(modelClass=RandomForestClassifier,
                       params=rf_params,
                       tuned_params=rf_tune_params,
                       isupdate=True, istune=True,
                       modelName='rf', cv=5)

lgb_params = {
    "boosting_type": "gbdt",
    "learning_rate": 0.1,  # shrinkage
    # "max_depth": 8,
    "num_leaves": 2**9,
    # "n_estimators": 1000,
    "subsample": 0.7,
    "colsample_bytree": 1,
    "seed": SEED,
    'n_jobs': -1,
    'silent': 1,
}

lgb_tune_params = {
    "learning_rate": [0.02, 0.2, 0.5, 0.8],  # shrinkage
    "max_depth": [8, 10, 15, 20],
    "subsample": [0.3, 0.5, 0.6, 0.8],
    'reg_lambda': [0.01, 0.2, 0.5, 0.8],
    "colsample_bytree": [0.3, 0.5, 0.7, 0.8],
    # "seed": [1, 2, 3, 4]
}

lgbtrainer_params = dict(modelClass=LGBMClassifier,
                       params=lgb_params, tuned_params=lgb_tune_params,
                       isupdate=True, istune=True,
                       modelName='lgb', cv=5)

extra_params = {
    'random_state': SEED
}

extra_tune_params = {
    'n_estimators': [50, 70, 90],
    'learning_rate': [0.5, 0.7, 0.9],
}

extratrainer_params = dict(modelClass=ExtraTreesClassifier,
                         params=extra_params, tuned_params=extra_tune_params,
                         isupdate=True, istune=True,
                         modelName='extra', cv=5)
adb_params = {
    'random_state': SEED
}
adb_tune_params = {
    'n_estimators': [50, 70, 90],
    'learning_rate': [0.5, 0.7, 0.9],
}
adbtrainer_params = dict(modelClass=AdaBoostClassifier,
                       params=adb_params, tuned_params=adb_tune_params,
                       isupdate=True, istune=True,
                       modelName='adb', cv=5)


trainers = []
adbtrainer = TrainerWrapper(baseTrainer, adbtrainer_params)
trainers.append(adbtrainer)

#extratrainer= TrainerWrapper(baseTrainer, extratrainer_params)
rftrainer = TrainerWrapper(ortTrainer, rftrainer_params)
trainers.append(rftrainer)

lgbtrainer = XgBoostWrapper(ortTrainer, lgbtrainer_params)
trainers.append(lgbtrainer)

features = [get_oof(trainer, X_train, y_train, X_test) for trainer in trainers]

X_train_new = np.hstack(feature[0] for feature in features)
X_test_new = np.hstack(feature[1] for feature in features)

finaltrainer = ortTrainer(modelClass=LGBMClassifier,
                        params=lgb_params, tuned_params=lgb_tune_params,
                        isupdate=True, istune=True,
                        modelName='final', cv=5)

finaltrainer.fit(X_train_new, y_train)

_, X_val_new, _, y_val_new = train_test_split(X_train_new, y_train, random_state=SEED)
finaltrainer.getOutput(X_test_new, test_id, X_val_new, y_val_new)

print("haha")
