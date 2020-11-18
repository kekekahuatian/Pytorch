import torch
import torchvision
import argparse
import cv2
import numpy as np

from commom import cocoName
import random


def random_color():
    b = random.randint(0, 255)
    g = random.randint(0, 255)
    r = random.randint(0, 255)

    return b, g, r


def main():
    imagePath = 'I:/EnglishCharacter/Recognition/test/img/word_1.png'
    input = []
    num_classes = 91
    names = cocoName.names

    print("Creating model")
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(num_classes=num_classes, pretrained=True)
    model = model.cuda()

    model.eval()

    src_img = cv2.imread(imagePath)
    img = cv2.cvtColor(src_img, cv2.COLOR_BGR2RGB)
    img_tensor = torch.from_numpy(img / 255.).permute(2, 0, 1).float().cuda()
    input.append(img_tensor)
    out = model(input)
    boxes = out[0]['boxes']
    labels = out[0]['labels']
    scores = out[0]['scores']

    for idx in range(boxes.shape[0]):
        if scores[idx] >= 0.5:
            # 锚箱坐标
            x1, y1, x2, y2 = int(boxes[idx][0]), int(boxes[idx][1]), int(boxes[idx][2]), int(boxes[idx][3])
            name = names.get(str(labels[idx].item()))
            cv2.rectangle(src_img, (x1, y1), (x2, y2), random_color(), thickness=2)
            cv2.putText(src_img, text=name, org=(x1, y1 + 10), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5, thickness=1, lineType=cv2.LINE_AA, color=(0, 0, 255))
    cv2.imshow('result', src_img)
    cv2.waitKey()
    cv2.destroyAllWindows()


main()
