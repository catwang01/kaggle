from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.externals import joblib
from my_ort import ORT
from utils import plotAuc, getNextVer
import time
import pandas as pd
import numpy as np
import re
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_curve, auc
from path import *

pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 60)


class Trainer:
    def __init__(self, modelClass,
                 params=None, tuned_params=None, modelName=0,
                 istune=False, isupdate=False, cv=5,
                 test_size=0.05, dataPath=processedDataPath):
        self.dataPath = dataPath
        self.test_size = test_size
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        self.modelClass = modelClass
        self.params = params
        self.test = None
        self.test_id = None
        self.model = modelClass(**params)
        self.cv = cv
        self.feature_names = None
        self.other_params = params
        self.tuned_params = tuned_params
        self.modelName = modelName
        version = getNextVer('{}-.*-(\d+).model'.format(modelName))
        self.isupdate = True if version == 1 else isupdate

        if self.dataPath.startswith("pca"):
            self.modelType = 'pca'
        elif self.dataPath.startswith('rein'):
            self.modelType = 'reinbalance'
        else:
            self.modelType = 'ordinal'

        if self.isupdate:
            self.modelName = '-'.join([modelName, self.modelType, str(version)])
        else:
            self.modelName = '-'.join([modelName, self.modelType, str(version - 1)])
        self.trainAuc = None
        self.valAuc = None

        if tuned_params is None:
            self.istune = False
        else:
            self.istune = istune

    def read_data(self):
        data = np.load(self.dataPath)
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(data['X'], data['y'], random_state=SEED,
                                                                              test_size=self.test_size)
        self.test = data['X_test']
        self.test_id = data['test_id']
        self.feature_names = data['feature_names']

    def fit(self, *args, **kwargs):
        if self.isupdate:
            if self.istune:
                self.tune(*args, **kwargs)
            else:
                self._fit(*args, **kwargs)
        else:
            self.model = joblib.load(os.path.join(root, self.modelName + '.model'))
        return self.model

    def _fit(self, *args, **kwargs):
        print("==================================================")
        print("Train Model {}".format(self.modelName))
        self.model.fit(self.X_train, self.y_train, *args, **kwargs)

        y_train_hat = self.model.predict_proba(self.X_train)[:, 1]
        y_val_hat = self.model.predict_proba(self.X_val)[:, 1]

        self.trainAuc = plotAuc(self.y_train, y_train_hat, "train")
        self.valAuc = plotAuc(self.y_val, y_val_hat, "val")

        print("Train auc: {} Val auc: {}".format(self.trainAuc, self.valAuc))
        joblib.dump(self.model, "{}".format(self.modelName + '.model'))
        print("Save Model {}".format(self.modelName))
        return self.model

    def tune(self, *args, **kwargs):

        for paramName in self.tuned_params:
            print("==================================================")
            print('Tuning param:{} values: {}'.format(paramName, self.tuned_params[paramName]))
            clf = GridSearchCV(self.model, {paramName: self.tuned_params[paramName]}, scoring='roc_auc', cv=self.cv,
                               verbose=1, n_jobs=-1)
            clf.fit(self.X_train, self.y_train)

            print("paramName: {} bestValue: {} TestScore: {} bestValScore; {}".format(
                paramName,
                clf.best_params_,
                plotAuc(self.y_train, clf.predict_proba(self.X_train)[:, -1]),
                clf.best_score_
            ))
            self.model = clf.best_estimator_
            self.other_params.update(clf.best_params_)

        print("Save Tuned Model {}".format(self.modelName))
        joblib.dump(self.model, "{}".format(self.modelName + '.model'))

    def getOutput(self):
        output = pd.DataFrame({'id': self.test_id, 'prob': self.model.predict_proba(self.test)[:, 1]})

        if self.valAuc is None:
            self.valAuc = plotAuc(self.y_val, self.model.predict_proba(self.X_val)[:, 1])

        outputPath = os.path.join(root, 'output-{modelName}-{valAuc:.4f}.txt'.format(
            modelName=self.modelName,
            valAuc=self.valAuc))
        output.to_csv(outputPath, index=False, columns=None, header=False, sep='\t')
        print("output: " + outputPath)
        return output


# class xgbTrainer(Trainer):
#
#     def _fit(self):
#         print("==================================================")
#         print("Train Model {}".format(self.modelName))
#         self.model.fit(self.X_train, self.y_train,
#                        eval_set=[(self.X_val, self.y_val)],
#                        early_stopping_rounds=10,
#                        eval_metric="auc",
#                        verbose=True)
#
#         y_train_hat = self.model.predict_proba(self.X_train)[:, 1]
#         y_val_hat = self.model.predict_proba(self.X_val)[:, 1]
#
#         self.trainAuc = plotAuc(self.y_train, y_train_hat, "train")
#         self.valAuc = plotAuc(self.y_val, y_val_hat, "val")
#
#         print("Train auc: {} Val auc: {}".format(self.trainAuc, self.valAuc))
#         joblib.dump(self.model, "{}".format(self.modelName + '.model'))
#         print("Save Model {}".format(self.modelName))
#         return self.model
#
#
# class lgbTrainer(Trainer):
#
#     def _fit(self):
#         print("==================================================")
#         print("Train Model {}".format(self.modelName))
#         self.model.fit(self.X_train, self.y_train,
#                        eval_set=[(self.X_val, self.y_val)],
#                        early_stopping_rounds=10,
#                        eval_metric="auc",
#                        verbose=True)
#
#         y_train_hat = self.model.predict_proba(self.X_train)[:, 1]
#         y_val_hat = self.model.predict_proba(self.X_val)[:, 1]
#
#         self.trainAuc = plotAuc(self.y_train, y_train_hat, "train")
#         self.valAuc = plotAuc(self.y_val, y_val_hat, "val")
#
#         print("Train auc: {} Val auc: {}".format(self.trainAuc, self.valAuc))
#         joblib.dump(self.model, "{}".format(self.modelName + '.model'))
#         print("Save Model {}".format(self.modelName))
#         return self.model
#

class ortTrainer(Trainer):

    def __init__(self, modelClass,
                 params=None, tuned_params=None, modelName=0,
                 istune=False, isupdate=False, cv=5,
                 test_size=0.05, dataPath=processedDataPath):

        super(ortTrainer, self).__init__(modelClass,
                                         params=params, tuned_params=tuned_params,
                                         modelName=modelName,
                                         istune=istune, isupdate=isupdate,
                                         cv=cv,
                                         test_size=test_size,
                                         dataPath=dataPath)

        # 构建正交实验表
        self.ort = ORT()

        # 获取参数正交表
        self.param_df = self.ort.genSets(self.tuned_params, mode=2)
        print(self.param_df)
        self.ort_res = None
        self.bestParams = None

    def tune(self, *args, **kwargs):
        df_params = self.param_df
        cur = 0
        total = len(df_params.index)

        ort_res = []
        ort_res_val = []
        ort_train_time = []

        params = self.params.copy()

        for i in df_params.index:
            cur += 1

            tmp = {}
            for j in range(df_params.shape[1]):
                dtype = df_params.dtypes[j]
                colName = df_params.columns[j]
                tmp[colName] = np.array([df_params.iloc[i, j]]).astype(dtype)[0]

            params.update(tmp)
            model = self.modelClass(**params)
            time_start = time.time()
            model.fit(self.X_train, self.y_train, *args, **kwargs)
            time_elapsed = int(time.time() - time_start)
            fpr, tpr, _ = roc_curve(self.y_train, model.predict_proba(self.X_train)[:, 1])
            res = auc(fpr, tpr)

            fpr, tpr, _ = roc_curve(self.y_val, model.predict_proba(self.X_val)[:, 1])
            res_val = auc(fpr, tpr)
            print(tmp)
            print('res: {res:.4f}, res_val: {res_val:.4f} , time: {time}, num: {cur}/{total}'.format(res=res,
                                                                                                     res_val=res_val,
                                                                                                     time=time_elapsed,
                                                                                                     cur = cur,
                                                                                                     total=total))
            ort_res.append(res)
            ort_res_val.append(res_val)
            ort_train_time.append(time_elapsed)

        self.ort_res = df_params
        self.ort_res['res'] = ort_res
        self.ort_res['res_val'] = ort_res_val
        self.ort_res['train_time'] = ort_train_time

        # 筛选最优值
        self.bestParams = dict(zip(
            self.tuned_params.keys(),
            map(lambda x: self.ort_res.groupby(x).res_val.mean().idxmax(), self.tuned_params.keys())
        ))

        self.model = self.modelClass(**self.bestParams)
        self.model.fit(self.X_train, self.y_train, *args, **kwargs)
        return self.model
