import torch
import torch.nn as nn
from torch.nn.functional import relu

class MLP(nn.Module):
    def __init__(self):
        super(MLP,self).__init__()
        self.fc1=nn.Linear(400,256)
        self.fc2=nn.Linear(256,128)
        self.fc3=nn.Linear(128,4)
        #self.fc4=nn.Linear(64,4)
        self.dropout=nn.Dropout(0.5)
    def forward(self,x):
        x=relu(self.fc1(x))
        x=self.dropout(x)
        x=relu(self.fc2(x))
        x=self.dropout(x)
        x=self.fc3(x)
        return x