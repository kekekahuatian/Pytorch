import torch
from torch.utils.data import Dataset
import pandas as pd
import torchvision.datasets as datasets
import torchvision.models as models
from torchvision import transforms as transforms


# 定义一个数据集
class BulldozerDataset(Dataset):
    """ 数据集演示 """

    def __init__(self, csv_file):
        """实现初始化方法，在初始化的时候将数据读载入"""
        self.df = pd.read_csv(csv_file)

    def __len__(self):
        """
        返回df的长度
        """
        return len(self.df)

    def __getitem__(self, idx):
        """
        根据 idx 返回一行数据
        """
        return self.df.iloc[idx].SalePrice


ds_demo = BulldozerDataset("data/median_benchmark.csv")
# print(len(ds_demo))
# print(ds_demo[0])

"""""
torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, sampler=None, num_workers=0, collate_fn=<function default_collate>, pin_memory=False, drop_last=False)
dataset (Dataset) – 加载数据的数据集。
batch_size (int, optional) – 每个batch加载多少个样本(默认: 1)。
shuffle (bool, optional) – 设置为True时会在每个epoch重新打乱数据(默认: False).
sampler (Sampler, optional) – 定义从数据集中提取样本的策略。如果指定，则忽略shuffle参数。
num_workers (int, optional) – 用多少个子进程加载数据。0表示数据将在主进程中加载(默认: 0)
collate_fn (callable, optional) –
pin_memory (bool, optional) –
drop_last (bool, optional) – 如果数据集大小不能被batch size整除，则设置为True后可删除最后一个不完整的batch。如果设为False并且数据集的大小不能被batch size整除，则最后一个batch将更小。(默认: False)
"""
dl = torch.utils.data.DataLoader(ds_demo, batch_size=10, shuffle=True, num_workers=0)
# iter 生成迭代器
idata = iter(dl)
# for i, data in enumerate(dl):
#     print(i, data)
'''
torchvision.datasets 可以理解为PyTorch团队自定义的dataset，
这些dataset帮我们提前处理好了很多的图片数据集，我们拿来就可以直接使用.

torchvision.models
torchvision不仅提供了常用图片数据集，还提供了训练好的模型，可以加载之后，
直接使用，或者在进行迁移学习 torchvision.models模块的 子模块中包含以下模型结构。

'''
trainset = datasets.MNIST(root='./data',  # 表示 MNIST 数据的加载的目录
                          train=True,  # 表示是否加载数据库的训练集，false的时候加载测试集
                          download=True,  # 表示是否自动下载 MNIST 数据集
                          transform=None)  # 表示是否需要对数据进行预处理，none为不进行预处理
# 我们直接可以使用训练好的模型，当然这个与datasets相同，都是需要从服务器下载的
resnet18 = models.resnet18(pretrained=True)

# transforms 模块提供了一般的图像转换操作类，用作数据处理和数据增强
transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),  # 先四周填充0，在把图像随机裁剪成32*32
    transforms.RandomHorizontalFlip(),  # 图像一半的概率翻转，一半的概率不翻转
    transforms.RandomRotation((-45, 45)),  # 随机旋转
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.229, 0.224, 0.225)),  # R,G,B每层的归一化用到的均值和方差
])
