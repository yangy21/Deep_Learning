import torch
from torch import nn
import torch.nn.functional as F
import pandas as pd
import numpy as np

from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

data = pd.read_csv('HR.csv')
data = data.join(pd.get_dummies(data.part)).join(pd.get_dummies(data.salary))

#print(data.head())
data.drop(columns=['part', 'salary'], inplace=True)

Y_data = data.left.values.reshape(-1, 1)
Y = torch.from_numpy(Y_data).type(torch.FloatTensor)
print('Y.shape:', Y.shape)

X_data = data[[c for c in data.columns if c != 'left']].values
X = torch.from_numpy(X_data).type(torch.FloatTensor)
print('X.shape:', X.shape)
print('----------')

HRdataset = TensorDataset(X, Y)
print('len(HRdataset):', len(HRdataset))
#print('HRdataset[2:5]:', HRdataset[2:5])

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.liner_1 = nn.Linear(20, 64)
        self.liner_2 = nn.Linear(64, 64)
        self.liner_3 = nn.Linear(64, 1)

    def forward(self, input):
        x = F.relu(self.liner_1(input))
        x = F.relu(self.liner_2(x))
        x = F.sigmoid(self.liner_3(x))
        return x

# model = Model()
# print(model)

lr = 0.0001

def get_model():
    model = Model()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    return model, opt


model, optim = get_model()
loss_fn = nn.BCELoss()
epochs = 10

batch = 64
no_of_batches = len(data) // batch

print('----------dataloader类-----------------')

HR_ds = TensorDataset(X, Y)
HR_dl = DataLoader(HR_ds, batch_size=batch, shuffle=True)


for epoch in range(epochs):
    for x, y in HR_dl:
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        optim.zero_grad()
        loss.backward()
        optim.step()
    with torch.no_grad():
        print('epoch:',epoch,'loss:',loss_fn(model(X), Y).data.item())






