import random
import os
import time

from PIL import Image, ImageDraw, ImageFont
from matplotlib import pyplot as plt

random.seed(time.time())
path_img = "I:/EnglishCharacter/Character/"
characterList = "abcdefghijklnmopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"


# 在目录下生成用来存放数字 1-9 的 9个文件夹，分别用 1-9 命名
def mkdir_for_imgs():
    for i in range(65, 91):  # chr(i)返回对应的ASC2字符w
        if os.path.isdir(path_img + "大" + chr(i)):
            pass
        else:
            print(path_img + "大" + chr(i))
            os.mkdir(path_img + "大" + chr(i))
    for i in range(97, 123):  # chr(i)返回对应的ASC2字符w
        if os.path.isdir(path_img + "小" + chr(i)):
            pass
        else:
            print(path_img + "小" + chr(i))
            os.mkdir(path_img + "小" + chr(i))


# 删除路径下的图片
def del_imgs():
    # for i in range(1, 10):
    #     dir_nums = os.listdir(path_img + "Num_" + str(i))
    #     for tmp_img in dir_nums:
    #         if tmp_img in dir_nums:
    #             # print("delete: ", tmp_img)
    #             os.remove(path_img + "Num_" + str(i) + "/" + tmp_img)

    for i in range(65, 91):  # chr(i)返回对应的ASC2字符
        dir_nums = os.listdir(path_img + "大" + chr(i))
        for tmp_img in dir_nums:
            if tmp_img in dir_nums:
                print("delete: ", tmp_img)
                os.remove(path_img + "大" + chr(i) + "/" + tmp_img)
    for i in range(97, 123):  # chr(i)返回对应的ASC2字符w
        dir_nums = os.listdir(path_img + "小" + chr(i))
        for tmp_img in dir_nums:
            if tmp_img in dir_nums:
                print("delete: ", tmp_img)
                os.remove(path_img + "小" + chr(i) + "/" + tmp_img)

    print("Delete finish", "\n")


# 生成单张扭曲的数字图像
def generate_single():
    # 先绘制一个50*50的空图像
    im_50_blank = Image.new('RGB', (50, 50), (255, 255, 255))

    # 创建画笔
    draw = ImageDraw.Draw(im_50_blank)

    # 生成随机字符

    num = random.randint(0, len(characterList) - 1)
    text = characterList[num]
    # 设置字体，这里选取字体大小25
    font = ImageFont.truetype('simsun.ttc', 25)

    # xy是左上角开始的位置坐标
    draw.text(xy=(18, 11), font=font, text=text, fill=(0, 0, 0))

    # 随机旋转-10-10角度
    random_angle = random.randint(-10, 10)
    im_50_rotated = im_50_blank.rotate(random_angle)

    # 图形扭曲参数
    params = [1 - float(random.randint(1, 2)) / 100,
              0,
              0,
              0,
              1 - float(random.randint(1, 10)) / 100,
              float(random.randint(1, 2)) / 500,
              0.001,
              float(random.randint(1, 2)) / 500]

    # 创建扭曲
    im_50_transformed = im_50_rotated.transform((50, 50), Image.PERSPECTIVE, params)

    # 生成新的30*30空白图像
    im_30 = im_50_transformed.crop([15, 15, 50, 50])
    # plt.figure()
    # plt.imshow(im_30)
    # plt.show()
    return im_30, text


# 生成手写体英文字母
def creatCharacter(n):
    # 用cnt_num[1]-cnt_num[9]来计数数字1-9生成的个数，方便之后进行命名
    cnt_num = []
    for i in range(52):
        cnt_num.append(0)
    for m in range(1, n + 1):
        # 调用生成图像文件函数
        im, generate_num = generate_single()

        # 取灰度
        im_gray = im.convert('1')

        # 统计个数,用来命名图像文件


        # 大写
        for j in range(65, 91):
            if generate_num == chr(j):
                cnt_num[j - 65] = cnt_num[j - 65] + 1
                # 路径如 "F:/code/***/P_generate_handwritten_number/data_pngs/1/1_231.png"
                # 输出显示路径
                print("Generate:", path_img + "大" + chr(j) + "/" + chr(j) + "_" + str(cnt_num[j - 65]) + ".png")
                # 将图像保存在指定文件夹中
                im_gray.save(path_img + "大" + chr(j) + "/" + chr(j) + "_" + str(cnt_num[j - 65]) + ".png")
        # 小写
        for j in range(97, 123):
            if generate_num == chr(j):
                cnt_num[j - 71] = cnt_num[j - 71] + 1
                # 路径如 "F:/code/***/P_generate_handwritten_number/data_pngs/1/1_231.png"
                # 输出显示路径
                print("Generate:", path_img + "小" + chr(j) + "/" + chr(j) + "_" + str(cnt_num[j - 71]) + ".png")
                # 将图像保存在指定文件夹中
                im_gray.save(path_img + "小" + chr(j) + "/" + chr(j) + "_" + str(cnt_num[j - 71]) + ".png")

    print("\n")

    print("生成的A~z的分布：")
    for k in range(52):
        print(characterList[k], ":", cnt_num[k], "in all")


# mkdir_for_imgs()
# del_imgs()
creatCharacter(100000)
