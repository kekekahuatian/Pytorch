import torch
from torch.utils.data import Dataset
from torchvision import models
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
import torch.optim as optim
import ImgModify
from torchvision.transforms import transforms
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

BatchSize = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 5
model = models.vgg16(pretrained=True)

trainPath = "I:/EnglishCharacter/Recognition/train/"
testPath = "I:/EnglishCharacter/Recognition/test/"
allTrainData, allTestData = ImgModify.getImgByPath(trainPath), ImgModify.getImgByPath(testPath)
tranData, testData = allTrainData[0], allTestData[0]
trainLabel, testLabel = allTrainData[1], allTestData[1]
tranData, testData = ImgModify.modifyData(tranData, 224, 224), ImgModify.modifyData(testData, 224, 224)
tranData = torch.utils.data.DataLoader(tranData, batch_size=BatchSize, shuffle=True)
testData = torch.utils.data.DataLoader(testData, batch_size=BatchSize, shuffle=True)
# for batch in tranData:
#     for data in batch:
#         data=data.view(500,500,3)
#         plt.imshow(data)
#         plt.show()

cost = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())


def train(model, device, trainLoader, trainLabel, optimizer, cost, epoch):
    model.eval()
    model.to(device)
    batchNum = 0
    for batch in trainLoader:
        optimizer.zero_grad()
        y = model(batch.to(device))
        loss = cost(y, trainLabel)
        loss.backward()
        optimizer.step()
        batchNum += 1
        # if batchNum%100==0:
        # print('EPOCH:{} loss:{} correct:{}%'.format(epoch,loss,)


train(model, DEVICE, tranData, trainLabel, optim, cost, EPOCHS)
print("debug")
