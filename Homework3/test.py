import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

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


class PoetryModel(nn.Module):
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
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# 设置超参数
model_path = 'model.pth'        # 模型路径
start_words = '黑云压城城欲摧'  # 唐诗的第一句
max_gen_len = 125                # 生成唐诗的最长长度


def generate(start_words, ix2word, word2ix):

    # 读取模型
    model = PoetryModel(len(word2ix), embedding_dim, hidden_dim)
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    
    # 读取唐诗的第一句
    results = list(start_words)
    start_word_len = len(start_words)
    
    # 设置第一个词为<START>
    input = torch.Tensor([word2ix['<START>']]).view(1, 1).long()
    input = input.to(device)
    hidden = None

    # 生成唐诗
    for i in range(max_gen_len):
        output, hidden = model(input, hidden)
        # 读取第一句
        if i < start_word_len:
            w = results[i]
            input = input.data.new([word2ix[w]]).view(1, 1)
        # 生成后面的句子
        else:
            top_index = output.data[0].topk(1)[1][0].item()
            w = ix2word[top_index]
            results.append(w)
            input = input.data.new([top_index]).view(1, 1)
        # 结束标志
        if w == '<EOP>':
            del results[-1]
            break
            
    return results

results = generate(start_words, ix2word, word2ix)
poetry = str()
for i in results:
    poetry += i
    if i == '。':
        poetry += '\n'
    
print(poetry)
