from baseModel import ortTrainer, baseTrainer
from path import *
from utils import load_data
from sklearn.linear_model import LogisticRegression

other_params = {
    'n_jobs': 1,
    'random_state': 1,
    'C': 0.3
}

tuned_params = {
    'C': [0.01, 0.05, 0.1, 0.3, 0.5, 0.7]
}

X_train, y_train, X_test, test_id, feature_names = load_data(processedDataPath)
trainer = baseTrainer(modelClass=LogisticRegression,
                      params=other_params,
                      tuned_params=tuned_params,
                      isupdate=True,
                      istune=True,
                      modelName='lr')

trainer.fit(X_train, y_train)
trainer.getOutput()
