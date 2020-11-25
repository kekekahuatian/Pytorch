import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader
from torchvision import models
from torchvision.transforms import transforms

BatchSize = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 5
model = models.vgg16(pretrained=True)
for parmaeter in model.parameters():
    parmaeter.requires_grad = False
model.classifier = torch.nn.Sequential(torch.nn.Linear(25088, 4096),
                                       torch.nn.ReLU(),
                                       torch.nn.Dropout(p=0.5),
                                       torch.nn.Linear(4096, 4096),
                                       torch.nn.ReLU(),
                                       torch.nn.Dropout(p=0.5),
                                       torch.nn.Linear(4096, 52))

trainPath = "I:/EnglishCharacter/character/"
testPath = "I:/EnglishCharacter/character/"
trainDataset = torchvision.datasets.ImageFolder(root=os.path.join(trainPath, 'train'),
                                                transform=transforms.Compose(
                                                    [
                                                        transforms.CenterCrop(224),
                                                        transforms.RandomHorizontalFlip(),
                                                        transforms.ToTensor(),
                                                        transforms.Normalize(
                                                            mean=(0.485, 0.456, 0.406),
                                                            std=(0.229, 0.224, 0.225))
                                                    ]))
testDataset = torchvision.datasets.ImageFolder(root=os.path.join(testPath, 'test'),
                                               transform=transforms.Compose(
                                                   [
                                                       transforms.CenterCrop(224),
                                                       transforms.RandomHorizontalFlip(),
                                                       transforms.ToTensor(),
                                                       transforms.Normalize(
                                                           mean=(0.485, 0.456, 0.406),
                                                           std=(0.229, 0.224, 0.225))
                                                   ]))

# 预览dataset  permute()交换shape
# for i in trainDataset:
#     plt.imshow(i[0].permute(1, 2, 0))
#     plt.show()
trainDataLoader = DataLoader(dataset=trainDataset, batch_size=BatchSize, shuffle=True)
testDataLoader = DataLoader(dataset=testDataset, batch_size=BatchSize, shuffle=True)
cost = nn.CrossEntropyLoss()
# optimizer = optim.SGD(model.parameters(), 0.05)
optimizer = optim.Adam(model.parameters())


def train(model, device, trainLoader, optimizer, cost, epoch):
    model.eval()
    model.to(device)
    batchNum = 0
    correct = 0
    start = time.time()
    for (data, label) in trainLoader:
        data, label = data.to(device), label.to(device)
        optimizer.zero_grad()
        y = model(data)
        loss = cost(y, label)
        loss.backward()
        optimizer.step()
        batchNum += 1
        pred = y.max(1, keepdim=True)[1]
        correct += pred.eq(label.view_as(pred)).sum().item()
        if batchNum % 300 == 0:
            print('EPOCH:{} loss:{:.4f} time:{:.2f}'.format(epoch, loss, -start + time.time()))
            start = time.time()
    print('EPOCH:{} correct:{:.4f}%'.format(epoch, 100 * correct / (BatchSize * batchNum)))


def test(model, device, testLoader, epoch):
    model.eval()
    model.to(device)
    correct = 0
    batchNum = 0
    for (data, label) in testLoader:
        batchNum += 1
        data, label = data.to(device), label.to(device)
        y = model(data)
        pred = y.max(1, keepdim=True)[1]
        correct += pred.eq(label.view_as(pred)).sum().item()
    print('TEST EPOCH:{} correct:{:.4f}%'.format(epoch, 100 * correct / (BatchSize * batchNum)))


for i in range(EPOCHS):
    start = time.time()
    train(model, DEVICE, trainDataLoader, optimizer, cost, i + 1)
    test(model, DEVICE, testDataLoader, i + 1)
    print((-start + time.time()) / 60)
