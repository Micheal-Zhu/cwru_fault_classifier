import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset

# __len__ 这样就可以通过len(dataset)返回数据集的大小。
# __getitem__ 支持索引，以便dataset[i]可以用来获取样本i
class GetDataset(Dataset):
    def __init__(self,model=0, train=True,  transform=None):
        if train:
            tsv_file = "你的cwru数据集路径/TRAIN.tsv"
        
        else:
            tsv_file = "你的cwru数据集路径/TEST.tsv"
        self.data_frame = pd.read_csv(tsv_file, sep = '\t', header=None)
        self.transform = transform
        self.model = model

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        series = self.data_frame.iloc[idx-1][1:]
        series = np.array(series)
        series = series.astype(np.float32)
        series = torch.from_numpy(series)
        label = self.data_frame.iloc[idx-1][0]
        label = int(label) 

        if self.transform:
            series = self.transform(series)

        if self.model!=2:
            return series.unsqueeze(-1),label
        elif self.model==2:
            return series, label
