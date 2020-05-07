from baseModel import Trainer
from sklearn.ensemble import RandomForestClassifier
from matplotlib import  pyplot as plt
from path import *
import numpy as np

other_params = {
    'n_jobs': -1,
    'n_estimators': 30,
    'max_depth': 12,
    'random_state': 2
}

tuned_params = {
    # 'n_estimators': [10, 15, 20, 25, 30],
    # 'max_depth': [10, 15, 20, 25]
    'random_state': [1, 2, 3, 4, 5]
}

trainer = Trainer(modelClass=RandomForestClassifier,
                  other_params=other_params,
                  tuned_params=tuned_params,
                  isupdate=True,
                  istune=True,
                  modelName='rf',
                  dataPath=processedDataPath)

trainer.read_data()
trainer.fit()
trainer.getOutput()

sorted_importances, sorted_featurenames= zip(*sorted(zip(trainer.model.feature_importances_, trainer.feature_names), reverse=True))
plt.plot(sorted_importances)
plt.title("变量累计贡献")
plt.show()
plt.plot(np.cumsum(sorted_importances))
plt.title("累计贡献")
plt.show()
sorted_featurenames[:10]
sorted_featurenames

