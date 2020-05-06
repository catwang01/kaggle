from sklearn.externals import joblib
from sklearn.ensemble import RandomForestClassifier
from baseModel import Trainer
import json
import pandas as pd
from path import *
from preprocess import plotAuc

x = joblib.load("haha.model")

train = pd.read_pickle(processed_train_tag_feature_path)

with open(jsonPath) as f:
    features, targets = json.load(f)

plotAuc(train[targets], x.predict_proba(train[features])[:, 1])

other_params = {
    'n_jobs': -1,
    'n_estimators': 10,
    'max_depth': 10

}

tuned_params = {
    'n_estimators': [10, 15, 20, 25, 30],
    'max_depth': [10, 15, 20, 25]
}

trainer1 = Trainer(
    modelClass=RandomForestClassifier,
    other_params=other_params,
    tuned_params=tuned_params,
    isupdate=False,
    isresample=False,
    istune=False,
    modelName='rf')

trainer1.read_data()
trainer1.fit()
trainer1.getOutput()

import numpy as np
df = pd.DataFrame({'col':[1,2,'3']})

def test(df):
    df['col'].replace({'3': 2}, inplace=True)

test(df)
df.dtypes

