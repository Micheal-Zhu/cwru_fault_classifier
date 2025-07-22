import torch
import torch.nn as nn
import math

dropout_value=0.1
class PositionalEncoding(nn.Module):
    #位置编码类
    def __init__(self, d_model, dropout=0.1, max_len=1000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 初始化位置编码矩阵 [max_len, d_model]
        pe = torch.zeros(max_len, d_model)

        # 生成位置序列 [0, 1, 2, ..., max_len-1]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # 计算除数项：10000^(2i/d_model)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() *
            (-math.log(10000.0) / d_model)
        )

        # 填充位置编码矩阵（正弦偶数列，余弦奇数列）
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数列
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数列

        # 增加batch维度 [1, max_len, d_model]
        pe = pe.unsqueeze(0)

        # 注册为缓冲区（不参与学习但保存到模型状态）
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: [batch_size, seq_len, d_model]
        x = x + self.pe[:, :x.size(1), :]  # 自动广播相加
        return self.dropout(x)

class Transformer_classifier(nn.Module):
    def __init__(self,input_size=1,d_model=8,n_head=4,num_layers=2,seq_len=400,dropout=0.0):
        super(Transformer_classifier, self).__init__()
        self.embedding = nn.Linear(input_size,d_model)

        self.positional_embedding = PositionalEncoding(d_model)

        self.encoder_layer=nn.TransformerEncoderLayer(d_model=d_model, nhead=n_head)
        self.transformer_encoder=nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.classifier=nn.Sequential(
            nn.Linear(d_model*seq_len, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 4)
        )
    def forward(self,x):
        x=self.embedding(x)
        x=self.positional_embedding(x)
        x=self.transformer_encoder(x)

        x=x.reshape(x.size(0),-1)#(batch_size, seq_len, d_model) -> (batch_size, seq_len*d_model),-1表示自动计算剩下的

        x=self.classifier(x)
        return x

