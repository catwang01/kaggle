import numpy as np
import re
import json
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb


dataPath = '~/kaggle/rossmann-store-sales'
trainDataPath = os.path.join(dataPath, 'train.csv')
testDataPath = os.path.join(dataPath, 'test.csv')
storeDataPath = os.path.join(dataPath, 'store.csv')
trainModelPath = 'trainFeatures.pkl'
testModelPath = 'testFeatures.pkl'
jsonModelPath = 'columnNames.json'

def getLatestVersion(pattern):
    fileVersion = 0
    for file in os.listdir():
        ret = re.search(pattern, file)
        if ret is not None:
            fileVersion = max(fileVersion, int(ret.group(1)))
    return fileVersion


def rmspe(y, yhat):
    return np.sqrt(np.mean((yhat / y - 1) ** 2))

def rmspe_xgb(pred, dtrain):
    yhat = dtrain.get_label()
    y = np.expm1(pred)
    yhat = np.expm1(yhat)
    return 'rmspe', rmspe(y, yhat)

def featureEncoder(df: pd.DataFrame, features: list) -> None:
    for feature in features:
        uniqvals = df[feature].unique()
        mappings = dict(zip(uniqvals, range(len(uniqvals))))
        df[feature].replace(mappings, inplace=True)

def build_features() -> tuple:
    if not os.path.exists(trainModelPath):
        return _build_features()
    else:
        with open(jsonModelPath) as f:
            [featureCol, targetCol] = json.load(f)

        train = pd.read_pickle(trainModelPath)
        test = pd.read_pickle(testModelPath)
        return train, test, featureCol, targetCol

def _build_features() -> tuple:
    featureCol, targetCol = [], []
    types = {
        'StateHoliday': str
    }

    train = pd.read_csv(trainDataPath, parse_dates=[2], dtype=types)
    test = pd.read_csv(testDataPath, parse_dates=[3], dtype=types)
    store = pd.read_csv(storeDataPath)

    ## Open
    train['Open'].value_counts()
    train = train[train['Open'] == 1]
    test['Open'].fillna(1, inplace=True)

    ## Sales 这个是target
    targetCol.append("Sales")

    ## Date
    featureCol.extend(['year', 'month', 'day', 'dayofweek', 'weekofyear'])

    for data in [train, test]:
        data['year'] = data['Date'].dt.year
        data['month'] = data['Date'].dt.month ### 几月
        data['day'] = data['Date'].dt.day ### 几号
        data['dayofweek'] = data['Date'].dt.dayofweek.value_counts() ## 星期几
        data['weekofyear'] = data['Date'].dt.weekofyear ## 第几周

    ## DayOfWeek
    ### 这个和 date 日期是重复，可以drop掉
    train.DayOfWeek.value_counts()
    train.drop(columns=['DayOfWeek'], inplace=True)
    test.drop(columns=['DayOfWeek'], inplace=True)

    ## Customers test没有 Customer这个要如何使用
    train['Customers'].value_counts(dropna=False)
    # test['Customers'].value_counts(dropna=False)
    # featureCol.append("Customers")

    ## Store
    train.Store.value_counts()
    featureCol.append("Store")

    ## Promo
    train['Promo'].value_counts()
    test['Promo'].value_counts()
    featureCol.append("Promo")

    ## StateHoliday
    train.StateHoliday.value_counts()
    test.StateHoliday.value_counts()
    for data in [train, test]:
        featureEncoder(data, features=['StateHoliday'])
    featureCol.append("StateHoliday")

    ## SchoolHoliday
    train['SchoolHoliday'].value_counts()
    test['SchoolHoliday'].value_counts()
    featureCol.append("SchoolHoliday")


    ## StoreType
    store['StoreType'].value_counts()
    featureEncoder(store, ['StoreType'])
    featureCol.append("StoreType")

    ## Assortment
    store['Assortment'].value_counts()
    featureEncoder(store, ['Assortment'])
    featureCol.append("Assortment")

    ## CompetitionDistance
    x = store['CompetitionDistance']
    x.isna().sum()
    x.fillna(x.mean(), inplace=True)
    featureCol.append("CompetitionDistance")

    ## CompetitionOpenSinceMonth
    x = store['CompetitionOpenSinceMonth']
    x.value_counts(dropna=False)
    featureCol.append("CompetitionOpenSinceMonth")

    ## CompetitionOpenSinceYear
    x = store['CompetitionOpenSinceYear']
    x.value_counts(dropna=False)
    featureCol.append("CompetitionOpenSinceYear")

    ## Promo2
    x = store['Promo2']
    x.value_counts(dropna=False)
    featureCol.append("Promo2")

    ## Promo2SinceWeek
    x = store['Promo2SinceWeek']
    x.value_counts(dropna=False)
    featureCol.append("Promo2SinceWeek")

    ## Promo2SinceYear
    x = store['Promo2SinceYear']
    x.value_counts(dropna=False)
    featureCol.append("Promo2SinceYear")

    ## PromoInterval
    x = store['PromoInterval']
    x.value_counts(dropna=False)
    featureEncoder(store, features=['PromoInterval'])
    featureCol.append("PromoInterval")

    train = pd.merge(train, store, on='Store')
    test = pd.merge(test, store, on='Store')

    train.to_pickle(trainModelPath)
    test.to_pickle(testModelPath)
    with open('columnNames.json', 'w') as f:
        json.dump([featureCol, targetCol], f)
    return train, test, featureCol, targetCol

params = {
    "objective": "reg:linear",
    "booster" : "gbtree",
    "eta": 0.3, # shrinkage
    "max_depth": 10,
    "subsample": 0.9,
    "colsample_bytree": 0.7,
    "silent": 1,
    "seed": 1301
}

num_boost_round = 3000

train, test, featureCol, targetCol = build_features()
print("Train a XGBoost model")
X_train, X_val, y_train, y_val = train_test_split(train[featureCol], train[targetCol], test_size=0.012, random_state=10)

y_train = np.log1p(y_train)
y_val = np.log1p(y_val)

dtrain = xgb.DMatrix(X_train, y_train)
dval = xgb.DMatrix(X_val, y_val)

dtest = xgb.DMatrix(test[featureCol])
watchlist = [(dtrain, 'train'), (dval, 'val')]
gbm = xgb.train(params, dtrain, num_boost_round, evals=watchlist, early_stopping_rounds=100, verbose_eval=True, feval=rmspe_xgb)
y_val_hat = gbm.predict(dval)


# save model
gbm.save_model('xgb{}.model'.format(1 + getLatestVersion(r'xgb(\d).model')))


ytest = np.expm1(gbm.predict(dtest))
ytest = np.round(ytest).astype(int)

mask = test['Open']==0
assert mask.sum() == 5984
output = pd.DataFrame({"Id": test['Id'], "Sales": ytest})
output.loc[mask, 'Sales'] = 0
assert (output['Sales']==0).sum() == mask.sum()

output.to_csv('submission{}.csv'.format(getLatestVersion(r'submission(\d).csv')+1), index=None)


