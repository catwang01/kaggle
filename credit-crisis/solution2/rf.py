from baseModel import Trainer
from sklearn.ensemble import RandomForestClassifier

other_params = {
    'n_jobs': -1,
    'n_estimators': 10,
    'max_depth': 10

}

tuned_params = {
    'n_estimators': [10, 15, 20, 25, 30],
    'max_depth': [10, 15, 20, 25]
}

trainer = Trainer(modelClass=RandomForestClassifier,
                  other_params=other_params,
                  tuned_params=tuned_params,
                  isupdate=False,
                  isresample=False,
                  istune=False,
                  modelName='rf',)

trainer.read_data()
trainer.fit()
trainer.getOutput()

