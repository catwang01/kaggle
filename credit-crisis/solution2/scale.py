import numpy as np
from sklearn.preprocessing import StandardScaler
from path import *

data = np.load(processedDataPath)
newdata = {
    'y': data['y'],
    'test_id': data['test_id'],
    'feature_names': data['feature_names']
}

scaler = StandardScaler()
newdata['X'] = scaler.fit_transform(data['X'])
newdata['X_test'] = scaler.transform(data['X_test'])

np.savez_compressed(scaledDataPath, **newdata)
