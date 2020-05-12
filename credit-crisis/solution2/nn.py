import torch
from utils import load_data
from path import *
from sklearn.model_selection import  train_test_split
from matplotlib import pyplot as plt

class Net(torch.nn.Module):
    def __init__(self, input_size):
        super(Net, self).__init__()
        self.fc = torch.nn.Sequential(
            torch.nn.BatchNorm1d(input_size, affine=False),
            torch.nn.Linear(input_size, 30),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(30, affine=False),
            torch.nn.Linear(30, 20),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(20, affine=False),
            torch.nn.Linear(20, 30),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(30, affine=False),
            torch.nn.Linear(30, input_size),
        )

    def forward(self, x):
        return self.fc(x)

X_train, y_train, X_test, test_id, feature_names = load_data(processedDataPath)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, random_state=SEED, test_size=TEST_SIZE)

x_train = torch.tensor(X_train, dtype=torch.float)
x_val = torch.tensor(X_val, dtype=torch.float)
y = torch.tensor(y_train, dtype=torch.float)

net = Net(x_train.shape[1])

mse = torch.nn.MSELoss()
optimizer = torch.optim.SGD(net.parameters(), lr=3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

def train_or_eval(net, x, stage):
    if stage == 'train':
        net.train()
        xhat = net(x)
        loss = mse(x, xhat)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    else:
        net.eval()
        xhat = net(x)
        loss = mse(x, xhat)
        scheduler.step(loss.item())

    print("Loss {stage}: {loss}".format(stage=stage, loss=loss.item()))
    print()
    return loss.item()

n_iters = 100000
loss_vals = []
print("Start training!")

for i in range(n_iters):
    print("Epoch: {}/{}".format(i, n_iters))
    print('-' * 10)

    loss_val = train_or_eval(net, x_train, 'train')
    loss_vals.append(loss_val)

    if i % 50 == 0:
        train_or_eval(net, x_val, 'val')

plt.plot(loss_vals)

PATH = 'model.pth'
torch.save(net.state_dict(), PATH)
