from sklearn.metrics import roc_curve, auc
import numpy as np
from sklearn.externals import  joblib
import pandas as pd
import matplotlib.pyplot as plt
import json
from path import root
import os
import re

def load_data(trainPath, testPath, jsonPath):
    train = pd.read_pickle(trainPath )
    test = pd.read_pickle(testPath)
    with open(jsonPath) as f:
        features, target = json.load(f)
    return train, test, features, target

def plotAuc(y, yhat, xlabel=None):
    fpr, tpr, thresholds = roc_curve(y, yhat, pos_label=1)
    plt.plot(fpr, tpr)
    myauc = auc(fpr, tpr)
    plt.title("auc: {}".format(myauc))
    if xlabel: plt.xlabel(xlabel)
    plt.show()
    return myauc

def getFreq(df):
    tmp = df.value_counts()
    return tmp / tmp.sum()

def isOutiler(df) -> bool:
    # 下四分位数值、中位数，上四分位数值
    Q1, median, Q3 = np.percentile(df, (25, 50, 75), interpolation='midpoint')
    # 四分位距
    IQR = Q3 - Q1

    # 内限
    inner = [Q1 - 1.5 * IQR, Q3 + 1.5 * IQR]
    # 外限
    outer = [Q1 - 3.0 * IQR, Q3 + 3.0 * IQR]
    print('>>>内限：', inner)
    print('>>>外限：', outer)

    # 过滤掉极端异常值
    return (df < outer[0]) | (df > outer[1])


def fillnaInplace(df, val):
    df.fillna(val, inplace=True)


def dropInplaceByCondition(df, condition):
    df.drop(df.index[condition], inplace=True)

def dropColumnInplace(df, columns):
    if isinstance(columns, str):
        columns = [columns]
    df.drop(columns=columns, inplace=True)

def trainModel(model, X_train, y_train, X_val, y_val, modelName=None, number=0):

    print("==================================================")
    print("Train Model {}-{}".format(modelName, number))
    model.fit(X_train, y_train)

    y_train_hat = model.predict_proba(X_train)[:, 1]
    y_val_hat = model.predict_proba(X_val)[:, 1]

    trainAuc = plotAuc(y_train, y_train_hat, "train")
    valAuc = plotAuc(y_val, y_val_hat, "val")
    print("Train auc: {} Val auc: {}".format(trainAuc, valAuc))

    print("Save Model {}-{}".format(modelName, number))
    joblib.dump(model, '{name}-{number}-{valAuc}.model'.format(name=modelName, number=number, valAuc=valAuc))
    return model

def getLatestVer(pattern):
    for file in sorted(os.listdir(root), key=lambda x: os.path.getctime(x), reverse=True):
        # print(file)
        ret = re.search(pattern, file)
        if ret is not None:
            i = int(ret.group(1))
            return i
    return 0

def getNextVer(pattern):
    return getLatestVer(pattern) + 1


