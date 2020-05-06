import numpy as np
from sklearn.preprocessing import StandardScaler
from path import *

data = np.load(processedDataPath)
newdata = {
    'y': data['y'],
    'X_test': data['X_test'],
    'id': data['id']
}

scaler = StandardScaler()
newdata['X'] = scaler.fit_transform(data['X'])
newdata['X_test'] = scaler.transform(data['X_test'])

np.savez_compressed(scaledDataPath, **data)
