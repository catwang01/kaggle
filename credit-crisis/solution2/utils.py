from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
import time
import numpy as np
from path import *
from sklearn.externals import  joblib
import pandas as pd
import matplotlib.pyplot as plt
from path import root
import os
import re

def plotAuc(y, yhat, xlabel=None, verbose=True):
    fpr, tpr, thresholds = roc_curve(y, yhat, pos_label=1)
    myauc = auc(fpr, tpr)
    if verbose:
        plt.plot(fpr, tpr)
        plt.title("auc: {}".format(myauc))
        if xlabel: plt.xlabel(xlabel)
        plt.show()
    return myauc

def load_data(dataPath):
    data = np.load(dataPath)
    return data['X'], data['y'], data['X_test'], data['test_id'], data['feature_names']

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
        ret = re.search(pattern, file)
        if ret is not None:
            i = int(ret.group(1))
            return i
    return 0

def getNextVer(pattern):
    return getLatestVer(pattern) + 1


def timeit(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        val = func(*args, **kwargs)
        time_elapsed = time.time() - start
        print('func: {name} used time {h}:{m}:{s}'.format(name=func.__name__,
                                                          h=int(time_elapsed / 3600),
                                                          m = int(time_elapsed / 60 % 60),
                                                          s = int(time_elapsed % 3600)
                                                          ))
        return val
    return wrapper
