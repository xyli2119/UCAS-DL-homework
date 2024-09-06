# 引入使用到的包
import argparse                              # 设置命令行参数以及超参数
import torch 
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim                  # 优化器
from torchvision import datasets, transforms # 数据加载器
from torch.optim.lr_scheduler import StepLR  # 自动更新学习率
import time

# 定义网络结构
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1)
        self.dropout1 = nn.Dropout(0.25) # 丢弃一些数据以防止过拟合
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(in_features=64*12*12, out_features=128) # 由于经过一个最大池化层后进入全连接层，所以维度为64*24*24/2/2
        self.fc2 = nn.Linear(in_features=128, out_features=10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1) # 拉平数据以送入全连接层
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1) # softmax用于分类
        return output

# 定义训练方法
def train(model, data_loader, device, optimizer, epoch, arg):
    model.train()
    for batch_index, (data, target) in enumerate(data_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad() # 清除梯度    
        output = model(data)
        loss = F.nll_loss(output,target)
        loss.backward() 
        optimizer.step() # 更新网络参数
        if batch_index % arg.log_interval == 0:
            print('Train Epoch: {} [{}/{} \t({:.0f}%)]\t Loss: {:.6f}'.format(
                epoch, batch_index * len(data), len(data_loader.dataset),
                100. * batch_index / len(data_loader), loss.item()
            ))
            


# 定义测试方法
def test(model, data_loader, device):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad(): # 测试模式禁止梯度计算
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction = 'sum').item() # 计算每一个test_batch的总损失
            pred = output.argmax(dim = 1, keepdim = True) # 选出softmax层后概率最大的以对比是否预测正确
            correct += pred.eq(target.view_as(pred)).sum().item() # 将布尔值张量中的 True 统计求和，得到的就是预测正确的样本数量。
    
    test_loss /= len(data_loader.dataset)
    acc_percent = 100. * correct / len(data_loader.dataset)
    
    print('\nTest dataset: Average Loss: {:.6f}, Accuracy: {}/{} ({:.3f}%)\n'.format(
        test_loss, correct, len(data_loader.dataset), acc_percent
    ))        

    
# 主函数
def main():
    # 可调整的训练参数 在命令行使用python3 train.py -h 查看命令帮助以设置训练参数
    parser = argparse.ArgumentParser(description = 'LXY Handwritten numeral recongnition')
    parser.add_argument('--batch-size', type=int, default=64,
                        help = '训练数据的batch_size，默认值为64')
    parser.add_argument('--test-batch-size', type=int, default=1000,
                        help = '测试数据的batch_size，默认值为1000')
    parser.add_argument('-epoch', type=int, default=10,
                        help = '训练轮数，默认值为10')
    parser.add_argument('--learning-rate', type=float, default=1.0,
                        help = '学习率，默认值为1.0')
    parser.add_argument('--gamma', type=float, default=0.7,
                        help = '学习率动态调整的步幅，默认值为0.7')
    parser.add_argument('--log-interval', type=int, default=50,
                        help = '多少个batch输出一次训练过程中的准确率以及损失，默认值为10')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help = '是否保存模型，默认不保存，保存则使用--save-model')
    args = parser.parse_args() # 超参数对象实例化
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    torch.manual_seed(5) # 设置随机数种子为5，以确保每次运行程序时的随机数序列相同
    
    train_kwargs = {'batch_size': args.batch_size} # 训练数据加载器的参数
    test_kwargs = {'batch_size': args.test_batch_size} # 测试数据加载器的参数
    if torch.cuda.is_available():
        print('GPU is ready.')
        time.sleep(2)
        GPU_kwargs = {'num_workers': 1, # 数据加载的子进程的数量
                       'pin_memory': True, # 存入GPU内存加速CUDA计算
                       'shuffle': True} # 数据打乱
        train_kwargs.update(GPU_kwargs)
        test_kwargs.update(GPU_kwargs)
    
    
    
    # 载入训练数据 
    # 首先定义transforms
    transform = transforms.Compose([
        transforms.ToTensor(), # 转换为tensor
        transforms.Normalize((0.1307,),(0.3081,)) #正则化，参数采用官方给出的参数
    ])
    # 加载数据
    train_data = datasets.MNIST(root = './data', transform = transform, download=True, train=True)
    test_data = datasets.MNIST(root = './data',transform = transform, train = False)
    
    train_loader = torch.utils.data.DataLoader(train_data, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(test_data, **test_kwargs)
    
    # 实例化网络结构
    model = Net().to(device)
    
    # 优化器
    optimizer = optim.Adadelta(model.parameters(), lr= args.learning_rate)
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)  # 动态调整学习率，加速收敛或者防止陷入局部最优解
    
    # 训练以及测试
    for epoch in range(1, args.epoch + 1):
        train(model,train_loader,device,optimizer,epoch,args)
        test(model,test_loader,device)
        scheduler.step() # 更新学习率
    
    # 保存模型
    if args.save_model:
        torch.save(model.state_dict, './mnist_cnn.pt')
        print("save model successfully!")
        
if __name__ == '__main__':
    main()