from baseModel import Trainer
from sklearn.ensemble import RandomForestClassifier
from matplotlib import  pyplot as plt
from path import *
from scipy.stats import pearsonr
import pandas as pd
import numpy as np

other_params = {
    'n_jobs': -1,
    'n_estimators': 30,
    'max_depth': 12,
}

tuned_params = {
    'n_estimators': [10, 20, 30, 40, 50, 60],
    'max_depth': [10, 15, 20, 25, 30]
}

trainer = Trainer(modelClass=RandomForestClassifier,
                  other_params=other_params,
                  tuned_params=tuned_params,
                  isupdate=True,
                  istune=False,
                  modelName='rf',
                  dataPath=reinbalancedDataPath)


trainer.read_data()
trainer.fit()
output = trainer.getOutput()

bestResult = pd.read_csv('currentBest.txt', header=None, sep='\t')
bestResult.columns = ['id', 'prob']

pccs = pearsonr(output['prob'], bestResult['prob'])
print("Score prediction: {}".format(pccs))


sorted_importances, sorted_featurenames= zip(*sorted(zip(trainer.model.feature_importances_, trainer.feature_names), reverse=True))
plt.plot(sorted_importances)
plt.title("变量累计贡献")
plt.show()
plt.plot(np.cumsum(sorted_importances))
plt.title("累计贡献")
plt.show()
sorted_featurenames[:10]
sorted_featurenames
