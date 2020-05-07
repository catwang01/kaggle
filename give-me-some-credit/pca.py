from path import *
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

data = np.load(scaledDataPath)

newdata = {
    'y': data['y'],
    'id': data['id']
}
pca = PCA(n_components=0.999999)
newdata['X'] = pca.fit_transform(data['X'])
newdata['X_test']= pca.transform(data['X_test'])

plt.plot(pca.explained_variance_ratio_)
plt.show()

print("#components:", pca.n_components_)
print("variances: ", np.cumsum(pca.explained_variance_ratio_))

np.savez_compressed(pcaDataPath, **newdata)
print("{} saved!".format(os.path.basename(pcaDataPath)))
