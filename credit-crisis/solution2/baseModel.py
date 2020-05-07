import os
import json
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.externals import joblib
import pandas as pd
from utils import plotAuc, getNextVer
from path import *

###### tag

class Trainer:
    def __init__(self, modelClass,
                 other_params=None, tuned_params=None,
                 modelName=0,
                 istune=False, isupdate=False,
                 dataPath=processedDataPath):
        self.dataPath = dataPath
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        self.test = None
        self.test_id = None
        self.feature_names = None
        self.modelClass = modelClass
        self.other_params = other_params
        self.tuned_params = tuned_params
        self.modelName = modelName
        version = getNextVer('{}-(\d)-.*.model'.format(modelName))
        self.isupdate = True if version==1 else isupdate

        if self.dataPath.startswith("pca"):
            self.modelType = 'pca'
        elif self.dataPath.startswith('rein'):
            self.modelType = 'reinbalance'
        else:
            self.modelType = 'ordinal'

        if self.isupdate:
            self.modelName = '-'.join([modelName, self.modelType, str(version)])
        else:
            self.modelName = '-'.join([modelName, self.modelType, str(version-1)])
        self.trainAuc = None
        self.valAuc = None

        if tuned_params is None:
            self.istune = False
        else:
            self.istune = istune

    def read_data(self):
        data = np.load(self.dataPath)
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(data['X'], data['y'], random_state=1, test_size=0.2)
        self.test = data['X_test']
        self.test_id = data['id']
        self.feature_names = data['feature_names']

    def fit(self):
        if self.isupdate:
            if self.istune:
                self.tune()
            else:
                self._fit()
        else:
            self.model = joblib.load(os.path.join(root, self.modelName + '.model'))
        return self.model

    def _fit(self):
        self.model = self.modelClass(**self.other_params)
        print("==================================================")
        print("Train Model {}".format(self.modelName))
        self.model.fit(self.X_train, self.y_train)

        y_train_hat = self.model.predict_proba(self.X_train)[:, 1]
        y_val_hat = self.model.predict_proba(self.X_val)[:, 1]

        self.trainAuc = plotAuc(self.y_train, y_train_hat, "train")
        self.valAuc = plotAuc(self.y_val, y_val_hat, "val")

        print("Train auc: {} Val auc: {}".format(self.trainAuc, self.valAuc))
        joblib.dump(self.model, "{}".format(self.modelName + '.model'))
        print("Save Model {}".format(self.modelName))
        return self.model

    def tune(self):

        self.model = self.modelClass(**self.other_params)

        for paramName in self.tuned_params:
            print("==================================================")
            print('Tuning param:{} values: {}'.format(paramName, self.tuned_params[paramName]))
            clf = GridSearchCV(self.model, {paramName: self.tuned_params[paramName]}, scoring='roc_auc', cv=10, verbose=1, n_jobs=-1)
            clf.fit(self.X_train, self.y_train)

            print("paramName: {} bestValue: {} bestScore; {}".format(
                paramName,
                clf.best_params_,
                clf.best_score_
            ))
            self.model = clf.best_estimator_
            self.other_params.update(clf.best_params_)

        print("Save Tuned Model {}".format(self.modelName))
        joblib.dump(self.model, "{}".format(self.modelName + '.model'))

    def getOutput(self):
        if self.model is None:
            self.fit()
        output = pd.DataFrame({'id': self.test_id, 'prob': self.model.predict_proba(self.test)[:, 1]})

        if self.valAuc is None:
            self.valAuc = plotAuc(self.y_val, self.model.predict_proba(self.X_val)[:, 1])

        outputPath =  os.path.join(root, 'output-{modelName}-{valAuc:.4f}.txt'.format(
            modelName=self.modelName,
            valAuc=self.valAuc))
        output.to_csv(outputPath, index=False, columns=None, header=False, sep='\t' )
        print("output: " + outputPath)

