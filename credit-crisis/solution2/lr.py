from baseModel import Trainer
from sklearn.linear_model import LogisticRegression

other_params = {
    'n_jobs': -1,

}

tuned_params = {
}

trainer = Trainer(modelClass=LogisticRegression,
                  other_params=other_params,
                  tuned_params=tuned_params,
                  isupdate=False,
                  isresample=False,
                  istune=False,
                  modelName='rf',)

trainer.read_data()
trainer.fit()
trainer.getOutput()

