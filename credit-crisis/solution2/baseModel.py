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
                 other_params, tuned_params,
                 modelName=0, isresample=False,
                 istune=False, isupdate=False):
        self.isresample = isresample
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        self.featureCols = None
        self.targetCols = None
        self.modelClass = modelClass
        self.other_params = other_params
        self.tuned_params = tuned_params
        self.modelName = modelName
        self.version = getNextVer('{}-(\d).model'.format(modelName))
        self.isupdate = True if self.version==1 else isupdate
        self.trainAuc = None
        self.valAuc = None
        self.istune = istune

    def read_data(self):
        self.read_train()


    def read_train(self):
        if self.isresample:
            train = pd.read_pickle(reinbalanced_data_path)
        else:
            train = pd.read_pickle(processed_train_tag_feature_path)

        with open(jsonPath) as f:
            self.featureCols, self.targetCols = json.load(f)

        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            train[self.featureCols],
            train[self.targetCols], test_size=0.05, random_state=10
        )

    def fit(self):
        if self.isupdate:
            if self.istune:
                self.tune()
            else:
                self._fit()
        else:
            self.model = joblib.load(os.path.join('{}-{}.model'.format(self.modelName, self.version-1)))
        return self.model

    def _fit(self):
        self.model = self.modelClass(**self.other_params)
        print("==================================================")
        print("Train Model {}-{}".format(self.modelName, self.version))
        self.model.fit(self.X_train, self.y_train)

        y_train_hat = self.model.predict_proba(self.X_train)[:, 1]
        y_val_hat = self.model.predict_proba(self.X_val)[:, 1]

        self.trainAuc = plotAuc(self.y_train, y_train_hat, "train")
        self.valAuc = plotAuc(self.y_val, y_val_hat, "val")

        print("Train auc: {} Val auc: {}".format(self.trainAuc, self.valAuc))
        joblib.dump(self.model, "{}-{}.model".format(self.modelName, self.version))
        print("Save Model {}-{}".format(self.modelName, self.version))
        return self.model

    def tune(self):

        self.model = self.modelClass(**self.other_params)

        for paramName in self.tuned_params:
            print("==================================================")
            print('Tuning param:{} values: {}'.format(paramName, self.tuned_params[paramName]))
            clf = GridSearchCV(self.model, {paramName: self.tuned_params[paramName]}, scoring='roc_auc', cv=10, verbose=1, n_jobs=-1)
            clf.fit(self.X_train.values, self.y_train.values.ravel())

            print("paramName: {} bestValue: {} bestScore; {}".format(
                paramName,
                clf.best_params_,
                clf.best_score_
            ))
            self.model = clf.best_estimator_
            self.other_params.update(clf.best_params_)

        print("Save Tuned Model {}-{}".format(self.modelName, self.version))
        joblib.dump(self.model, "{}-{}.model".format(self.modelName, self.version))

    def getOutput(self):
        if self.model is None:
            self.fit()
        test = pd.read_pickle(processed_test_tag_feature_path)
        output = pd.DataFrame({'id': test.id, 'prob': self.model.predict_proba(test[self.featureCols])[:, 1]})

        if self.valAuc is None:
            self.valAuc = plotAuc(self.y_val, self.model.predict_proba(self.X_val)[:, 1])

        outputPath =  os.path.join(root, 'output-{modelName}-{version}-{tune}-{valAuc:.4f}.txt'.format(
            modelName=self.modelName,
            version=getNextVer('output-{}-(\d+).*.txt'.format(self.modelName)),
            tune = 'resample' if self.isresample else 'ordinal',
            valAuc=self.valAuc))
        output.to_csv(outputPath, index=False, columns=None, header=False, sep='\t' )
        print("output: " + outputPath)




