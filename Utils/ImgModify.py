import os
import cv2
from torchvision.transforms import transforms


def getImgByPath(dataPath):
    """
    :param dataPath: 图片文件路径，里面只能有图片
    :return: 包含该文件内所有图片的列表
    """
    image = []
    label = []
    for files in os.walk(dataPath + "img/"):
        for file in files[2]:
            img = cv2.imread(dataPath + "img/" + file)
            image.append(img)
    file = open(dataPath + "label.txt", encoding='utf-8')
    for line in file:
        label.append(line)
    file.close()
    return [image, label]


def modifyData(dataList, h=0, w=0):
    """
    :param w: resize参数，不传就不resize
    :param h: resize参数，不传就不resize
    :param dataList: 图片列表
    :return: 归一化后的tensor数据
    """
    i = 0
    for data in dataList:
        if h & w != 0:
            modify = transforms.Compose([
                        transforms.ToPILImage(),
                        transforms.Resize((h, w)),
                        transforms.ToTensor()
                        ])
        else:
            modify = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataList[i] = modify(data)
        i += 1
    return dataList
