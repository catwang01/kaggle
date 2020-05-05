from xgboost import XGBClassifier
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
import pandas as pd
import os
from utils import load_data, plotAuc, getNextVer
from path import root, processed_train_trd_feature_path, \
    processed_test_trd_feature_path, \
    processed_train_tag_feature_path, \
    processed_test_tag_feature_path, \
    jsonPath
from test import SMOTE


def trainModel(model, X_train, y_train, X_val, y_val, modelName=None, number=None):

    if number is None:
        number = getNextVer('xgb-\d.model')
    print("Train Model {}-{}".format(modelName, number))
    model.fit(X_train, y_train,
                   eval_set=[(X_val, y_val)],
                   early_stopping_rounds=10,
                   eval_metric="auc",
                   verbose=True)

    print("Save Model {}-{}".format(modelName, number))
    joblib.dump(model, '{name}-{number}.model'.format(name=modelName, number=number))

    y_train_hat = model.predict_proba(X_train)[:, 1]
    y_val_hat = model.predict_proba(X_val)[:, 1]

    plotAuc(y_train, y_train_hat)
    plotAuc(y_val, y_val_hat)
    return model

################ tag

train_tag, test_tag, features, target = load_data(processed_train_tag_feature_path, processed_test_tag_feature_path,
                                                  jsonPath)

test_tag_id = test_tag.id
valid_features = list(set(train_tag.columns.values) & set(features))
test_tag = test_tag[valid_features]

imbalanced_arr = np.c_[train_tag[valid_features].values, train_tag[target].values]
print("Balancing!")
result = SMOTE(imbalanced_arr)
result = pd.DataFrame(result, columns=valid_features + ['flag'])
result.to_pickle("resampled_test_feature.pkl")

X_train_tag, X_val_tag, y_train_tag, y_val_tag = train_test_split(
    result[valid_features],
    result['flag'], test_size=0.05, random_state=10
)

# X_train_tag, X_val_tag, y_train_tag, y_val_tag = train_test_split(
#     train_tag[valid_features],
#     train_tag[target], test_size=0.05, random_state=10
# )

params = {
    'objective': 'binary:logistic',
    "booster": "gbtree",
    "eta": 0.01,  # shrinkage
    "max_depth": 8,
    "subsample": 0.7,
    "colsample_bytree": 0.6,
    "seed": 1301,
    "scale_pos_weight": 1
}
xgb_tag = trainModel(XGBClassifier(**params), X_train_tag, y_train_tag, X_val_tag, y_val_tag, modelName='xgb-tag')

output = pd.DataFrame({'id': test_tag_id, 'prob': xgb_tag.predict_proba(test_tag)[:, 1] })
output.to_csv(os.path.join(root, 'output-xgb-{}.txt'.format(getNextVer('output-xgb-(\d).txt'))), index=False, columns=None, header=False, sep='\t')
getNextVer('output-xgb-(\d).txt')


################ trd

# train_trd, test_trd, _, _ = load_data(processed_train_trd_feature_path, processed_test_trd_feature_path, jsonPath)
#
# test_trd_id = test_trd.id
# valid_features = list(set(train_trd.columns.values) & set(features))
# test_trd = test_trd[list(set(train_trd.columns.values) & set(features))]

# X_train_trd, X_val_trd, y_train_trd, y_val_trd = train_test_split(
#     train_trd[valid_features],
#     train_trd[target], test_size=0.05, random_state=10
# )
#
# params = {
#     'objective': 'binary:logistic',
#     "booster": "gbtree",
#     "eta": 0.01,  # shrinkage
#     "max_depth": 8,
#     "subsample": 0.7,
#     "colsample_bytree": 0.6,
#     "seed": 1301,
# }
# xgb_trd = trainModel(XGBClassifier(**params), X_train_trd, y_train_trd, X_val_trd, y_val_trd, modelName='xgb-trd')
#
# output_tag = pd.DataFrame({
#     'id': test_tag_id,
#     'prob': xgb_tag.predict_proba(test_tag)[:, 1]})
#
# output_trd = pd.DataFrame({
#     'id': test_trd_id,
#     'prob': xgb_trd.predict_proba(test_trd)[:, 1]})
#
# output_trd_indices = set(output_trd)
# ids, probs = [], []
# for id in output_tag.id:
#     ids.append(id)
#     if id in output_trd_indices:
#         probs.append(output_trd[output_trd.id==id]['prob'].values[0])
#     else:
#         probs.append(output_tag[output_tag.id==id]['prob'].values[0])
#
# output = pd.DataFrame({'id': ids, 'prob': probs})
# output.to_csv(os.path.join(root, 'output-xgb-{}.txt'.format(getNextVer('output-xgb-(\d).txt'))), index=False, columns=None, header=False, sep='\t')
#
# train_tag.flag.value_counts()
