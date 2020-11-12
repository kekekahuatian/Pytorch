import os

import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import models
from torchvision.transforms import transforms
import time

data_dir = './data/hymenopteraData'
batchSize = 8
EPOCHS = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 让torch判断是否使用GPU，建议使用GPU环境，因为会快很多
train_dataset = torchvision.datasets.ImageFolder(root=os.path.join(data_dir, 'train'),
                                                 transform=transforms.Compose(
                                                     [
                                                         transforms.RandomResizedCrop(224),  # 将原图随机裁剪成224*224
                                                         transforms.RandomHorizontalFlip(),  # 将原图随机翻转
                                                         transforms.ToTensor(),
                                                         transforms.Normalize(
                                                             mean=(0.485, 0.456, 0.406),
                                                             std=(0.229, 0.224, 0.225))
                                                     ]))

val_dataset = torchvision.datasets.ImageFolder(root=os.path.join(data_dir, 'val'),
                                               transform=transforms.Compose(
                                                   [
                                                       transforms.RandomResizedCrop(224),
                                                       transforms.RandomHorizontalFlip(),
                                                       transforms.ToTensor(),
                                                       transforms.Normalize(
                                                           mean=(0.485, 0.456, 0.406),
                                                           std=(0.229, 0.224, 0.225))
                                                   ]))

trainLoader = DataLoader(dataset=train_dataset, batch_size=batchSize, shuffle=True)
testLoader = DataLoader(dataset=val_dataset, batch_size=batchSize, shuffle=True)

model = models.vgg16(pretrained=True)

for parma in model.parameters():
    parma.requires_grad = False

model.classifier = torch.nn.Sequential(torch.nn.Linear(25088, 4096),
                                       torch.nn.ReLU(),
                                       torch.nn.Dropout(p=0.5),
                                       torch.nn.Linear(4096, 4096),
                                       torch.nn.ReLU(),
                                       torch.nn.Dropout(p=0.5),
                                       torch.nn.Linear(4096, 2))

model = model.to(DEVICE)
cost = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.classifier.parameters())


def train(model, device, trainLoader, optimizer, epoch):
    # enumerate函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标
    # 说人话就是让顺序排列的数据带上数字索引
    for batch_idx, (data, label) in enumerate(trainLoader):
        # 判断是否使用cuda
        data, label = data.to(device), label.to(device)
        # 梯度归零
        optimizer.zero_grad()
        # 正向传播
        output = model(data)
        # 计算损失
        loss = cost(output, label)
        # 反向传播
        loss.backward()
        # 更新参数
        optimizer.step()
        if (batch_idx + 1) % 30 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(trainLoader.dataset),
                       100. * batch_idx / len(trainLoader), loss.item()))


def test(model, device, testLoader):
    # 当模型使用了dropout和BN时一定要使用eval()来进行测试集的测试
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():  # 意思是不需要进行反向传播
        for data, target in testLoader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += cost(output, target)  # 将一批的损失相加
            pred = output.max(1, keepdim=True)[1]  # 找到概率最大的下标
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(testLoader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(testLoader.dataset),
        100. * correct / len(testLoader.dataset)))


for i in range(1, EPOCHS + 1):
    star = time.time()
    train(model, DEVICE, trainLoader, optimizer, i)
    test(model, DEVICE, testLoader)
    print((time.time() - star))
