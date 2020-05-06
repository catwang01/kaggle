import numpy as np
from sklearn.preprocessing import StandardScaler
from path import *

data = np.load(processedDataPath)

X_train, X_val, y_train, y_val, X_test = [
    data['X_train'],
    data['X_val'],
    data['y_train'],
    data['y_val'],
    data['X_test']
]

scaler = StandardScaler()

transformed_X_train = scaler.fit_transform(X_train)
transformed_X_val = scaler.fit_transform(X_val)
transformed_X_test = scaler.fit_transform(X_test)

data = {
    "X_train": transformed_X_train,
    "X_val": transformed_X_val,
    "y_train": y_train,
    "y_val": y_val,
    "X_test": transformed_X_test,
}
np.savez_compressed(scaledDataPath, **data)


