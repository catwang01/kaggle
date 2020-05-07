#!/usr/bin/env python

# coding: utf-8
import os
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from utils import plotAuc, getFreq, fillnaInplace, dropInplaceByCondition, dropColumnInplace, isOutiler
from path import *
import json
import numpy as np
import argparse

pd.set_option('display.max_rows', 10)
pd.set_option('display.max_columns', 10)

parser = argparse.ArgumentParser()
parser.add_argument("-v", "--verbose", type=bool, default=False, help="whether verbose or not")
args = parser.parse_args()

def printCount(df):
    if args.verbose:
        print(df.value_counts(dropna=False))
        print("dtype:", df.dtype)
        print("Nan ratio:", df.isna().sum() / df.shape[0] )


def processCs(data, features, mappings):
    for stage in ['train', 'test']:
        df = data[stage]['cs']

        printCount(df['MonthlyIncome'])
        fillnaInplace(df['MonthlyIncome'], df['MonthlyIncome'].mean())

        printCount(df['NumberOfDependents'])
        fillnaInplace(df['NumberOfDependents'], df['NumberOfDependents'].mean())


def castType(train, test, mappings):
    for dtype in mappings:
        for col in train.columns:
            if col in mappings[dtype]:
                if col == 'flag': continue
                if dtype == 'encode':
                    try:
                        encoder = LabelEncoder()
                        train[col] = encoder.fit_transform(train[col])
                        test[col] = encoder.transform(test[col])
                    except:
                        print("haha: encode " + col)
                else:
                    try:
                        train[col] = train[col].astype(dtype)
                        test[col] = test[col].astype(dtype)
                    except:
                       print("haha: " + col)

def process_data():

    data = {}

    data['train'] = {
        'cs':  pd.read_csv("/Users/ed/kaggle/give-me-some-credit/data/cs-training.csv", index_col=0)
    }

    data['test'] = {
        'cs': pd.read_csv("/Users/ed/kaggle/give-me-some-credit/data/cs-test.csv", index_col=0)
    }

    features = set(data['train']['cs'].columns.values)
    target = 'SeriousDlqin2yrs'
    features.remove(target)

    mappings = {
        'encode': [],
    }

    processors = [processCs]
    for processor in processors:
        processor(data, features, mappings)

    for k in mappings:
        mappings[k] = list(set(mappings[k]))

    features = list(features)

    train = data['train']['cs'].copy()
    y = train[target].values
    train = train[features]
    train.fillna(0, inplace=True)
    trainarray = train.values

    test = data['test']['cs'].copy()
    test_id = test.index.values
    test = test[features]
    assert train.shape[1] == test.shape[1]
    testarray = test.values

    data = {
        "X": trainarray,
        "y": y,
        "X_test": testarray,
        'feature_names': features,
        'test_id': test_id
    }
    np.savez_compressed(processedDataPath, **data)
    print('Export new files {}!'.format(processedDataPath))

if __name__ == '__main__':
    process_data()
