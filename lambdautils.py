import numpy as np
from torch import nn
import torch.nn.functional as F



class lambdanet(nn.Module):
    
    def __init__(self, input_dim):
        super(lambdanet, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.25),
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.25),
            nn.Linear(16, 1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.layers(x)

class lambdaset(Dataset):
    def __init__(self, X_train, X_test, y_train, y_test, train=True):

        if train:
            self.x_data, self.y_data = X_train, y_train
        else:
            self.x_data, self.y_data = X_test, y_test
    
    def __getitem__(self, i):
        return self.x_data[i], self.y_data[i], i

    def __len__(self):
        return self.y_data.shape[0]
