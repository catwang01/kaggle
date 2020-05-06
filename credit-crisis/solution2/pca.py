from path import *
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import json
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd

with open(jsonPath) as f:
    features, target = json.load(f)
train = pd.read_pickle(processed_train_tag_feature_path)
train, y = train[features], train[target]
test = pd.read_pickle(processed_test_tag_feature_path)

scaler = StandardScaler()
train = scaler.fit_transform(train[features])
test = scaler.transform(test[features])

pca = PCA(n_components=1)
train_transformed = pca.fit_transform(train)
test_transformed = pca.transform(test)

plt.plot(pca.explained_variance_ratio_)
plt.show()

print("#components:", pca.n_components_)
print("variances: ", np.cumsum(pca.explained_variance_ratio_))

X_train, X_val , y_train, y_val =  train_test_split(train, y.values.ravel())

data = {
    'X_train': X_train,
    'X_val': X_val,
    'y_train': y_train,
    'y_val': y_val,
    'X_test': test
}

np.savez_compressed(pcaDataPath, **data)
print("{} saved!".format(os.path.basename(pcaDataPath)))
