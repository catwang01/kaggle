from matplotlib import pyplot as plt
from utils import load_data
from sklearn.model_selection import train_test_split
from path import *
from sklearn.ensemble import RandomForestClassifier
from baseModel import ortTrainer
from scipy.stats import pearsonr
import pandas as pd
import numpy as np

other_params = {
    'n_jobs': -1,
    'n_estimators': 500,
    'max_depth': 3,
    'random_state': SEED
}

tuned_params = {
    'n_estimators': [400, 600, 800, 1000, 1200],
    'max_depth': [3, 4, 5, 6, 7],
    'random_state': [1, 2, 3, 4, 5]
}

X_train, y_train, X_test, test_id, feature_names = load_data(processedDataPath)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, random_state=SEED, test_size=TEST_SIZE)
trainer = ortTrainer(modelClass=RandomForestClassifier,
                     params=other_params,
                     tuned_params=tuned_params,
                     isupdate=True, istune=True,
                     modelName='rf', cv=5)

trainer.fit(X_train, y_train)
output = trainer.getOutput(X_test, test_id, X_val, y_val)

bestResult = pd.read_csv('currentBest.txt', header=None, sep='\t')
bestResult.columns = ['id', 'prob']

pccs = pearsonr(output['prob'], bestResult['prob'])
print("Score prediction: {}".format(pccs))
print(trainer.bestParams)

sorted_importances, sorted_featurenames = zip(
    *sorted(zip(trainer.model.feature_importances_, feature_names), reverse=True)
)

plt.plot(sorted_importances)
plt.title("变量累计贡献")
plt.show()
plt.plot(np.cumsum(sorted_importances))
plt.title("累计贡献")
plt.show()



