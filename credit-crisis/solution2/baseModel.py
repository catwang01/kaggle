from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score
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


class baseTrainer:
    def __init__(self, modelClass,
                 params=None, tuned_params=None, modelName=0,
                 istune=False, isupdate=False, cv=5):
        self.modelClass = modelClass
        self.params = params
        self.model = modelClass(**params)
        self.cv = cv
        self.feature_names = None
        self.other_params = params
        self.tuned_params = tuned_params
        self.modelName = modelName
        version = getNextVer('{}-.*-(\d+).model'.format(modelName))
        self.isupdate = True if version == 1 else isupdate

        if self.isupdate:
            self.modelName = '-'.join([modelName, str(version)])
        else:
            self.modelName = '-'.join([modelName, str(version - 1)])

        if tuned_params is None:
            self.istune = False
        else:
            self.istune = istune

    def fit(self, X, y, *args, **kwargs):
        if self.isupdate:
            if self.istune:
                self.tune(X, y, *args, **kwargs)
            else:
                self._fit(X, y, *args, **kwargs)
        else:
            self.model = joblib.load(os.path.join(root, self.modelName + '.model'))
        return self.model

    def predict(self, X):
        return self.model.predict_proba(X)[:, 1]

    def _fit(self, X, y, *args, **kwargs):
        X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=SEED, test_size=TEST_SIZE)
        print("==================================================")
        print("Train Model {}".format(self.modelName))
        self.model.fit(X_train, y_train, *args, **kwargs)

        y_train_hat = self.model.predict_proba(X_train)[:, 1]
        y_val_hat = self.model.predict_proba(X_val)[:, 1]

        trainAuc = roc_auc_score(y_train, y_train_hat)
        valAuc = roc_auc_score(y_val, y_val_hat)

        print("Train auc: {} Val auc: {}".format(trainAuc, valAuc))
        joblib.dump(self.model, "{}".format(self.modelName + '.model'))
        print("Save Model {}".format(self.modelName))
        return self.model

    def tune(self, X, y, *args, **kwargs):

        X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=SEED, test_size=TEST_SIZE)
        for paramName in self.tuned_params:
            print("==================================================")
            print('Tuning param:{} values: {}'.format(paramName, self.tuned_params[paramName]))
            clf = GridSearchCV(self.model, {paramName: self.tuned_params[paramName]}, scoring='roc_auc', cv=self.cv,
                               verbose=1, n_jobs=-1)
            clf.fit(X_train, y_train, *args, **kwargs)

            print("paramName: {} bestValue: {} TestScore: {} bestValScore; {}".format(
                paramName,
                clf.best_params_,
                plotAuc(y_train, clf.predict_proba(X_train)[:, -1]),
                clf.best_score_
            ))
            self.model = clf.best_estimator_
            self.other_params.update(clf.best_params_)

        print("Save Tuned Model {}".format(self.modelName))
        joblib.dump(self.model, "{}".format(self.modelName + '.model'))

    def getOutput(self, X_test, test_id, X_val, y_val):
        output = pd.DataFrame({'id': test_id, 'prob': self.predict(X_test)})

        valAuc = plotAuc(y_val, self.predict(X_val))

        outputPath = os.path.join(root, 'output-{modelName}-{valAuc:.4f}.txt'.format(
            modelName=self.modelName,
            valAuc=valAuc))
        output.to_csv(outputPath, index=False, columns=None, header=False, sep='\t')
        print("output: " + outputPath)
        return output


class ortTrainer(baseTrainer):

    def __init__(self, modelClass,
                 params=None, tuned_params=None, modelName=0,
                 istune=True, isupdate=False, cv=5):

        super(ortTrainer, self).__init__(modelClass, params=params, tuned_params=tuned_params,
                                         modelName=modelName, istune=istune, isupdate=isupdate, cv=cv)
        # 构建正交实验表
        self.ort = ORT()

        # 获取参数正交表
        self.param_df = self.ort.genSets(self.tuned_params, mode=2)

        print(self.param_df)
        self.ort_res = None
        self.bestParams = None

    def tune(self, X, y, *args, **kwargs):
        X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=SEED, test_size=TEST_SIZE)
        df_params = self.param_df.copy()
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
            model.fit(X_train, y_train, *args, **kwargs)
            time_elapsed = int(time.time() - time_start)
            res = roc_auc_score(y_train, model.predict_proba(X_train)[:, 1])
            res_val = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])

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
        self.model.fit(X_train, y_train, *args, **kwargs)
        return self.model
