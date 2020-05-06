from baseModel import Trainer
from path import *
from sklearn.linear_model import LogisticRegression

other_params = {
    'n_jobs': 1,
    'random_state': 1,
    'C': 0.3
}

tuned_params = {
    'C': [0.01, 0.05, 0.1, 0.3, 0.5, 0.7]
}

trainer = Trainer(modelClass=LogisticRegression,
                  other_params=other_params,
                  tuned_params=tuned_params,
                  isupdate=True,
                  istune=True,
                  modelName='lr',
                  dataPath=scaledDataPath)

trainer.read_data()
trainer.fit()
trainer.getOutput()
