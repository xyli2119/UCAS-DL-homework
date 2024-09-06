import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

def prepareData():
    
    # 读入预处理的数据
    datas = np.load("tang.npz",allow_pickle=True)
    data = datas['data']
    ix2word = datas['ix2word'].item()
    word2ix = datas['word2ix'].item()
    
    # 转为torch.Tensor
    data = torch.from_numpy(data)
    dataloader = DataLoader(data,
                         batch_size = 64,
                         shuffle = True,
                         num_workers = 2)
    
    return dataloader, ix2word, word2ix


dataloader, ix2word, word2ix = prepareData()


class PoetryModel(nn.Module): # 网络结构，其中由embedding层，LSTM层，线性层构成
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(PoetryModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, self.hidden_dim, num_layers=2)
        self.linear = nn.Linear(self.hidden_dim, vocab_size)

    def forward(self, input, hidden = None):
        seq_len, batch_size = input.size()
        
        if hidden is None:
            h_0 = input.data.new(2, batch_size, self.hidden_dim).fill_(0).float()
            c_0 = input.data.new(2, batch_size, self.hidden_dim).fill_(0).float()
        else:
            h_0, c_0 = hidden

        embeds = self.embedding(input)
        output, hidden = self.lstm(embeds, (h_0, c_0))
        output = self.linear(output.view(seq_len * batch_size, -1))
        return output, hidden
    
    
# 设置超参数
learning_rate = 5e-3       # 学习率
embedding_dim = 1024        # 嵌入层维度
hidden_dim = 256           # 隐藏层维度
model_path = None          # 预训练模型路径
epochs = 40                # 训练轮数,40epoch后过拟合
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def train(dataloader, ix2word, word2ix):

    # 配置模型，是否继续上一次的训练
    model = PoetryModel(len(word2ix), embedding_dim, hidden_dim)
    if model_path:
        model.load_state_dict(torch.load(model_path))
    model.to(device)
    
    # 设置优化器
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

    # 设置损失函数
    criterion = nn.CrossEntropyLoss()

    # 定义训练过程
    save_loss = 100.
    train_loss = list()
    train_loss.append(10.1)
    for epoch in range(epochs):
        loop = tqdm(enumerate(dataloader), total = len(dataloader),ncols=150)
        avg_loss = 0
        min_loss = 100.
        for (batch_idx, data) in loop:
            data = data.long().transpose(1, 0).contiguous()
            data = data.to(device)
            input, target = data[:-1, :], data[1:, :]
            output, _ = model(input)
            loss = criterion(output, target.view(-1))
            loop.set_description(f'Train Epoch [{epoch+1}/{epochs}][{batch_idx * len(data[1])}/{len(dataloader.dataset)}]')
            loop.set_postfix(loss = f"{loss.item():.4f}")
            lossnow = loss.item()
            avg_loss += lossnow
            if lossnow < min_loss:
                min_loss = lossnow
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        avg_loss = avg_loss/len(dataloader)
        train_loss.append(avg_loss)
        print('This epoch average train loss is {:.4f}, This epoch minimum train loss is {:.4f}'.format(avg_loss,min_loss))
        if avg_loss < save_loss:
            save_loss = avg_loss # 该循环loss小于目前最小loss，保存模型
            # 保存模型
            # print(save_loss)
            torch.save(model.state_dict(), 'model.pth')
            print("This epoch's model had been saved successfully.")
        else:
            print("no saving the model.")
            
            
    # 输出图像记录Loss        
    plt.plot(list(range(0, epochs+1)),train_loss, color='b', label='Train Loss')

    plt.ylabel('Loss', fontsize = 15)
    plt.yticks(fontsize=16)
    plt.xlabel('Epoch', fontsize = 15)
    plt.xticks(fontsize=16)
    plt.legend(fontsize=15, frameon=False)
    plt.title('Training Loss over Epochs',fontsize = 20)
    # plt.show()  # Option to view graph while training
    plt.savefig('graph_loss.png', bbox_inches='tight')
   # plt.close('all')



    
train(dataloader, ix2word, word2ix)
