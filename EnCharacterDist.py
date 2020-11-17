import torch
from torch.utils.data import Dataset
from torchvision import models

import ImgModify

model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
trainPath = "I:/EnglishCharacter/Recognition/train/"
testPath = "I:/EnglishCharacter/Recognition/test/"
BatchSize = 2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

allTrainData, allTestData = ImgModify.getImgByPath(trainPath), ImgModify.getImgByPath(testPath)
tranData, testData = allTrainData[0], allTestData[0]
trainLabel, testLabel = allTrainData[1], allTestData[1]
tranData, testData = ImgModify.modifyData(tranData), ImgModify.modifyData(testData)
tranData = torch.utils.data.DataLoader(tranData, batch_size=BatchSize, shuffle=True)
testData = torch.utils.data.DataLoader(testData, batch_size=BatchSize, shuffle=True)



print("debug")
