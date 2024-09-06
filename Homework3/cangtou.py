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
                         batch_size = 16,
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
verbose = True             # 打印训练过程
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# 设置超参数
model_path = 'model.pth'                 # 模型路径
start_words_acrostic = '黑云压城城欲摧'  # 唐诗的“头”
max_gen_len_acrostic = 125               # 生成唐诗的最长长度


def gen_acrostic(start_words, ix2word, word2ix):

    # 读取模型
    model = PoetryModel(len(word2ix), embedding_dim, hidden_dim)
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    
    # 读取唐诗的“头”
    results = []
    start_word_len = len(start_words)
    
    # 设置第一个词为<START>
    input = (torch.Tensor([word2ix['<START>']]).view(1, 1).long())
    input = input.to(device)
    hidden = None

    index = 0            # 指示已生成了多少句
    pre_word = '<START>' # 上一个词

    # 生成藏头诗
    for i in range(max_gen_len_acrostic):
        output, hidden = model(input, hidden)
        top_index = output.data[0].topk(1)[1][0].item()
        w = ix2word[top_index]

        # 如果遇到标志一句的结尾，喂入下一个“头”
        if (pre_word in {u'。', u'！', '<START>'}):
            # 如果生成的诗已经包含全部“头”，则结束
            if index == start_word_len:
                break
            # 把“头”作为输入喂入模型
            else:
                w = start_words[index]
                index += 1
                input = (input.data.new([word2ix[w]])).view(1, 1)
                
        # 否则，把上一次预测作为下一个词输入
        else:
            input = (input.data.new([word2ix[w]])).view(1, 1)
        results.append(w)
        pre_word = w
        
    return results

results_acrostic = gen_acrostic(start_words_acrostic, ix2word, word2ix)
poetry = str()
for i in results_acrostic:
    poetry += i
    if i == '。':
        poetry += '\n'
    
print(poetry)


