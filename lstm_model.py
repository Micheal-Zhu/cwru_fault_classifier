import torch
import torch.nn as nn
import torch.nn.functional as F



class LSTM_model(nn.Module):
    def __init__(self):
        super(LSTM_model, self).__init__()
        self.lstm=nn.LSTM(input_size=1,hidden_size=8,num_layers=1,batch_first=True)
        self.fc=nn.Linear(8,4)
        self.dropout=nn.Dropout(0.0)
    def forward(self,x):
        x,_=self.lstm(x)
        x=x[:,-1,:]
        x=self.dropout(x)
        x=self.fc(x)
        return x