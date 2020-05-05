from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import os
from utils import load_data, plotAuc, trainModel, getNextVer
from path import root, processed_train_trd_feature_path, \
    processed_test_trd_feature_path, \
    processed_train_tag_feature_path, \
    processed_test_tag_feature_path, \
    jsonPath

###### tag

train_tag, test_tag, features, target = load_data(processed_train_tag_feature_path, processed_test_tag_feature_path,
                                                  jsonPath)
test_tag_id = test_tag.id
test_tag = test_tag[list(set(train_tag.columns.values) & set(features))]
X_train_tag, X_val_tag, y_train_tag, y_val_tag = train_test_split(
    train_tag[list(set(train_tag.columns.values) & set(features))],
    train_tag[target], test_size=0.05, random_state=10
)

lr_tag = trainModel(LogisticRegression(verbose=True), X_train_tag, y_train_tag, X_val_tag, y_val_tag, modelName='lr', number=1)

####### trd

# train_trd, test_trd, _, _ = load_data(processed_train_trd_feature_path, processed_test_trd_feature_path, jsonPath)
# X_train_trd, X_val_trd, y_train_trd, y_val_trd = train_test_split(
#     train_trd[list(set(train_trd.columns.values) & set(features))],
#     train_trd[target], test_size=0.05, random_state=10
# )
#
# test_trd_id = test_trd.id
# test_trd = test_trd[list(set(train_trd.columns.values) & set(features))]
# lr_trd = trainModel(LogisticRegression(), X_train_trd, y_train_trd, X_val_trd, y_val_trd, modelName='lr', number=1)

output_tag = pd.DataFrame({
    'id': test_tag_id,
    'prob': lr_tag.predict_proba(test_tag)[:, 1]})

# output_trd = pd.DataFrame({
#     'id': test_trd_id,
#     'prob': lr_trd.predict_proba(test_trd)[:, 1]})
#
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
# output.to_csv(os.path.join(root, 'output-lr-{}.txt'.format(getNextVer('output-lr-(\d).txt'))), index=False, columns=None, header=False, sep='\t')
