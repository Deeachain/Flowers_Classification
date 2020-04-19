import os
import torch.utils.data as data
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as transforms
import glob

# 默认输入网络的图片大小
IMAGE_H = 200
IMAGE_W = 200

# 定义一个转换关系，用于将图像数据转换成PyTorch的Tensor形式
data_transform = transforms.Compose([
    transforms.ToTensor()   # 转换成Tensor形式，并且数值归一化到[0.0, 1.0]
])


class DogsVSCatsDataset(data.Dataset):      # 新建一个数据集类，并且需要继承PyTorch中的data.Dataset父类
    def __init__(self, mode, dir):          # 默认构造函数，传入数据集类别（训练或测试），以及数据集路径
        self.mode = mode
        self.list_imgdir = []                  # 新建一个image list，用于存放图片路径，注意是图片路径
        self.list_category = []
        self.list_img = []
        self.list_label = []                # 新建一个label list，用于存放图片对应猫或狗的标签，其中数值0表示猫，1表示狗
        self.data_size = 0                  # 记录数据集大小
        self.transform = data_transform     # 转换关系

        if self.mode == 'train':            # 训练集模式下，需要提取图片的路径和标签
            dir = dir + '/train/'           # 训练集路径在"dir"/train/
            for file in os.listdir(dir):    # 遍历dir文件夹
                self.list_imgdir.append(dir + file + '/')        # 将图片文件夹路径添加至image list
            for dirlist in self.list_imgdir:    # 遍历图片文件夹路径
                self.list_category.append(glob.glob(dirlist + '*.jpg'))
            for imglistpath  in self.list_category:  #每个类别的图片路径是一个列表
                for img in imglistpath:
                    self.data_size += 1      # 计算数据集数量
                    self.list_img.append(img)
                    category = img.split('/')[3]
                    #label采用one-hot编码，"0"表示daisy，"1"表示dandelion，在采用CrossEntropyLoss()计算Loss情况下，label只需要输入"1"的索引
                    if category == 'daisy':
                        self.list_label.append(0)         # 图片为daisy，label为0
                    elif category == 'dandelion':
                        self.list_label.append(1)         # 图片为dandelion，label为1，注意：list_img和list_label中的内容是一一配对的
                    elif category == 'rose':
                        self.list_label.append(2)         # 图片为rose，label为2，注意：list_img和list_label中的内容是一一配对的
                    elif category == 'sunflower':
                        self.list_label.append(3)         # 图片为sunflower，label为3，注意：list_img和list_label中的内容是一一配对的
                    elif category == 'tulip':
                        self.list_label.append(4)         # 图片为tulip，label为4，注意：list_img和list_label中的内容是一一配对的
        elif self.mode == 'test':           # 测试集模式下，只需要提取图片路径就行
            dir = dir + '/test/'            # 测试集路径为"dir"/test/
            for file in os.listdir(dir):
                self.list_img.append(dir + file)    # 添加图片路径至image list
                self.data_size += 1
            for img in self.list_img:
                category = img[16:17]           # 添加测试图片的文件名中的类别label
                self.list_label.append(category)
        else:
            return print('Undefined Dataset!')

    def __getitem__(self, item):            # 重载data.Dataset父类方法，获取数据集中数据内容
        if self.mode == 'train':                                        # 训练集模式下需要读取数据集的image和label
            img = Image.open(self.list_img[item])                       # 打开图片
            img = img.resize((IMAGE_H, IMAGE_W))                        # 将图片resize成统一大小
            img = np.array(img)[:, :, :3]                               # 数据转换成numpy数组形式
            label = self.list_label[item]                               # 获取image对应的label
            return self.transform(img), torch.LongTensor([label])       # 将image和label转换成PyTorch形式并返回
        elif self.mode == 'test':                                       # 测试集只需读取image
            img = Image.open(self.list_img[item])
            img = img.resize((IMAGE_H, IMAGE_W))
            img = np.array(img)[:, :, :3]
            label = self.list_label[item]
            return self.transform(img), label, self.list_img[item]       # 返回image和label
        else:
            print('None')

    def __len__(self):
        return self.data_size               # 返回数据集大小

