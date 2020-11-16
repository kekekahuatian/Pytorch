import torch
import torchvision
from torchvision import datasets, transforms, models
import os
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable
import time

path = "I:\dogs-vs-cats-redux-kernels-edition"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 10
BatchSize = 64
transform = transforms.Compose([transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

data_image = {x: datasets.ImageFolder(root=os.path.join(path, x),
                                      transform=transform)
              for x in ["train", "val"]}

data_loader_image = {x: torch.utils.data.DataLoader(dataset=data_image[x],
                                                    batch_size=BatchSize,
                                                    shuffle=True)
                     for x in ["train", "val"]}
# 引用vgg16的模型
model = models.vgg16(pretrained=True)
for parma in model.parameters():
    # 冻结参数，即不更新参数
    parma.requires_grad = False

# 对全连接层进行改写
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

#
# def train(model, device, trainLoader, optimizer, epoch):
#     # enumerate函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标
#     # 说人话就是让顺序排列的数据带上数字索引
#     for batch_idx, data in trainLoader:
#         # 判断是否使用cuda
#         X,y=data[0].to(device),data[1].to(device)
#         # 梯度归零
#         optimizer.zero_grad()
#         # 正向传播
#         print(data.size())
#         model.train=True
#         output = model(data)
#         # 计算损失
#         loss = cost(output, label)
#         # 反向传播
#         loss.backward()
#         # 更新参数
#         optimizer.step()
#         if (batch_idx + 1) % 500 == 0:
#             print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
#                 epoch, batch_idx * len(data), len(trainLoader.dataset),
#                        100. * batch_idx / len(trainLoader), loss.item()))
#
#
# def test(model, device, testLoader):
#     # 当模型使用了dropout和BN时一定要使用eval()来进行测试集的测试
#     model.eval()
#     test_loss = 0
#     correct = 0
#     with torch.no_grad():  # 意思是不需要进行反向传播
#         for data, target in testLoader:
#             data, target = data.to(device), target.to(device)
#             output = model(data)
#             test_loss += cost(output, target)  # 将一批的损失相加
#             pred = output.max(1, keepdim=True)[1]  # 找到概率最大的下标
#             correct += pred.eq(target.view_as(pred)).sum().item()
#     test_loss /= len(testLoader.dataset)
#     print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
#         test_loss, correct, len(testLoader.dataset),
#         100. * correct / len(testLoader.dataset)))
#
#
# for i in range(1, EPOCHS + 1):
#     star = time.time()
#     train(model, DEVICE, data_loader_image["train"], optimizer, i)
#     test(model, DEVICE, data_loader_image["val"])
#     print((time.time() - star))
for epoch in range(EPOCHS):
    since = time.time()
    print("Epoch{}/{}".format(epoch, EPOCHS))
    print("-" * 10)
    for param in ["train", "val"]:
        if param == "train":
            model.train = True
        else:
            model.train = False

        running_loss = 0.0
        running_correct = 0
        batch = 0
        for data in data_loader_image[param]:
            batch += 1

            X, y = data[0].to(DEVICE), data[1].to(DEVICE)
            optimizer.zero_grad()
            y_pred = model(X)
            _, pred = torch.max(y_pred.data, 1)

            loss = cost(y_pred, y)
            if param == "train":
                loss.backward()
                optimizer.step()
            running_loss += loss.data
            running_correct += torch.sum(pred == y.data)
            if batch % 100 == 0 and param == "train":
                print("Batch {}, Train Loss:{:.4f}, Train ACC:{:.4f}".format(
                    batch, running_loss / (BatchSize * batch), 100 * running_correct / (BatchSize * batch)))

        epoch_loss = running_loss / len(data_image[param])
        epoch_correct = 100 * running_correct / len(data_image[param])

        print("{}  Loss:{:.4f},  Correct{:.4f}".format(param, epoch_loss, epoch_correct))
    now_time = time.time() - since
    print("Training time is:{:.0f}m {:.0f}s".format(now_time // 60, now_time % 60))
