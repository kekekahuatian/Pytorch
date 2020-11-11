import torch.nn as nn  # 神经网络包
import torch.nn.functional as F  # 常用函数包，放了一些没有学习参数的函数
import torch.optim  # 优化器


# 神经网络结构
class Net(nn.Module):
    def __init__(self):
        # nn.Module子类的函数必须在构造函数中执行父类的构造函数
        super(Net, self).__init__()
        # 接下来定义网络
        # 卷积层 输入图片为通道， 输出通道数，卷积核
        self.conv1 = nn.Conv2d(1, 6, 3)
        # 线性层，输入特征，输出特征
        self.fc1 = nn.Linear(1350, 10)

    # 正向传播函数，Pytorch会根据forward函数自动进行反向传播
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)
        # 就是reshape函数 -1表示自适应
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        return x


net = Net()
input = torch.randn(1, 1, 32, 32)


out = net(input)
# arrange生成连续数 range同上当只返回list
y = torch.arange(0, 10).view(1, 10).float()
# 一定要先生成实例
lossFunction = nn.MSELoss()
loss = lossFunction(out, y)
# 新建一个优化器，SGD只需要要调整的参数和学习率
optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
# 先梯度清零(与net.zero_grad()效果一样)
# optimizer.zero_grad()
net.zero_grad()
loss.backward()
# 更新参数
optimizer.step()
