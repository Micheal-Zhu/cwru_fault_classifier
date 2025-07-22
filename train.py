import torch
import torch.nn as nn
from mlp import MLP
from GetDataset import GetDataset
from torch.utils.data import DataLoader
from lstm_model import LSTM_model
from transformer_model import Transformer_classifier
from data_increase import get_test_transform,get_train_transform
import matplotlib.pyplot as plt


#设置随机种子
def set_random_seed(seed):


    # 设置 PyTorch 的随机数种子
    torch.manual_seed(seed)
    # 如果使用 CUDA（GPU）
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)  # 设置当前 GPU 的随机数种子
        torch.cuda.manual_seed_all(seed)  # 设置所有 GPU 的随机数种子
        torch.backends.cudnn.deterministic = True  # 确保卷积操作是确定性的
        torch.backends.cudnn.benchmark = False  # 关闭 cuDNN 的自动优化

set_random_seed(12)



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
epochs=60
batch_size=16

use_tc=False#使用transformer_encoder+mlp分类的模型；需要在生成dataset的类中设置signals.unsqueeze(-1)来适配模型输入。
use_lstm=True#使用lstm+mlp分类的模型；需要在生成dataset的类中设置signals.unsqueeze(-1)来适配模型输入。
use_mlp=False#使用mlp模型；

if use_lstm:
    model_save_path = "./lstm_model.pt"
    model_num=0
elif use_tc:
    model_save_path = "./transformer_model.pt"
    model_num=1
elif use_mlp:
    model_save_path = './mlp_model.pt'
    model_num=2

#获取数据集。
train_processor=get_train_transform()
test_processor=get_test_transform()

train_dataset = GetDataset(model=model_num,train=True,transform=train_processor)
test_dataset = GetDataset(model=model_num,train=False,transform=test_processor)
train_loader=DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
test_loader=DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)


def train_model(model):

    train_losses,test_losses=[],[]


    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=3)

    # 训练循环
    for epoch in range(epochs):

        # 训练模式
        model.train()
        # 损失和准确度
        train_loss, correct, total = 0, 0, 0

        for signals, labels in train_loader:
            signals, labels = signals.to(device), labels.to(device)

            # if use_lstm or use_tc:
            #     signals=signals.unsqueeze(-1)#把维度从(batch_size,400)转为(batch_size,400,1)，因为lstm的输入维度为(batch_size,seq_len,input_size)
            #
            # 前向传播
            optimizer.zero_grad()#清空梯度
            outputs = model(signals)#前向传播
            loss = criterion(outputs, labels)#计算损失
            loss.backward()#反向传播一次计算梯度
            optimizer.step()#优化一次参数

            train_loss += loss.item()

            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        # 验证集评估
        val_acc,val_loss = evaluate(model, test_loader, device)#验证集评估，计算本轮正确率。

        scheduler.step(val_acc)#更新学习率

        print(f'Epoch {epoch + 1}: '
              f'Train Loss: {train_loss / len(train_loader):.4f}, '
              f'Train Acc: {100. * correct / total:.2f}%, '
              f'Val Acc: {100. * val_acc:.2f}%')


        train_losses.append(train_loss/len(train_loader))
        test_losses.append(val_loss)  # 记录验证损失

    torch.save(model.state_dict(),model_save_path)
    print("模型已成功保存在:"+model_save_path+"!")

    return train_losses,test_losses


def evaluate(model, loader, device):
    # 设置模型为评估模式，不进行梯度计算
    model.eval()
    correct, total = 0, 0  # 初始化正确预测数和总样本数计数器
    # 在此上下文中，PyTorch不会计算梯度，节省内存并加速验证过程
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        # 遍历数据加载器中的每一个批次
        num=0#记录batch数
        total_loss=0
        for signals, labels in loader:
            num+=1

            # 将输入数据转移到指定设备（GPU或CPU）
            outputs = model(signals.to(device))
            #计算loss
            total_loss += criterion(outputs, labels.to(device))
            # 获取模型输出中最大值的索引，即预测类别
            _, predicted = torch.max(outputs.data, 1)
            # 更新总样本数
            labels=labels.to(device)#也要将labels移动到同一设备上。
            total += labels.size(0)
            # 计算当前批次中正确预测的样本数，并累加到correct中
            correct += (predicted==labels).sum().item()#将数据移动到cpu上，然后将true求和取数量，用item()转换成python数据类型
    # 返回准确率：正确预测数 / 总样本数
    return (correct / total),(total_loss/num).item()


# 启动训练
if __name__ == "__main__":

    if use_lstm:
        model=LSTM_model().to(device)
        model_name="lstm"
        model_loss_path="./lstm_model_loss.png"
    elif use_tc:
        model=Transformer_classifier().to(device)
        model_name="transformer_encoder"
        model_loss_path="./transformer_model_loss.png"
    else:
        model=MLP().to(device)
        model_name="mlp"
        model_loss_path="./mlp_model_loss.png"

    # 训练模型
    train_losses,test_losses=train_model(model)
    plt.figure(figsize=(10,6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(test_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(model_name+'Training and Validation Loss')
    plt.legend()

    plt.savefig(model_loss_path)
    plt.show()

    # 最终测试
    test_acc,test_loss = evaluate(model, test_loader, device)
    print(f'Final Test Accuracy: {100. * test_acc:.2f}%')


    # with open(model_name+"-config.txt", "w", encoding="utf-8") as f:
    #     f.write("batch_size: "+str(batch_size))
