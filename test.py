from getdata import DogsVSCatsDataset as DVCD
from torch.utils.data import DataLoader as DataLoader
from models.resnet_all import *
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image


dataset_dir = './flowers'                   # 数据集路径
model_file = './checkpoint/model.pth'       # 模型保存路径
batch_size = 1                              # batch_size大小
workers = 6

def test():
    datafile = DVCD('test', dataset_dir)    # 实例化一个数据集
    dataloader = DataLoader(datafile, batch_size=batch_size, shuffle=True, num_workers=workers)

    model = resnet152(pretrained=True)                      # 实例化一个网络
    # 提取fc层中固定的参数
    fc_features = model.fc.in_features
    # 修改类别为5
    model.fc = nn.Linear(fc_features, 5)
    '''总共有5个类别的花，最后的fc层需要改成5'''

    model.cuda()                                        # 送入GPU，利用GPU计算
    model.load_state_dict(torch.load(model_file))       # 加载训练好的模型参数
    model.eval()                                        # 设定为评估模式，即计算过程中不要dropout

    print('Dataset loaded! length of train set is {0}'.format(len(datafile)))
    image_list = []
    label_list = []
    predict_list = []
    for img, label, name in dataloader:
        img = img.cuda()                                  # 将数据放置在PyTorch的Variable节点中，并送入GPU中作为网络计算起点
        out = model(img)                                            # 网路前向计算，输出图片属于
        print('predict is :', int(torch.argmax(out)))
        if int(torch.argmax(out)) == 0:
            print('the image is daisy flower')
        elif int(torch.argmax(out)) == 1:
            print('the image is dandelion flower')
        elif int(torch.argmax(out)) == 2:
            print('the image is rose flower')
        elif int(torch.argmax(out)) == 3:
            print('the image is sunflower flower')
        elif int(torch.argmax(out)) == 4:
            print('the image is tulip flower')
        print('预测图片是： ', name[0])
        image_list.append(str(name[0]))
        label_list.append(int(label[0]))
        predict_list.append(int(torch.argmax(out)))

    '''显示图片'''
    for i in range(len(image_list)):
        img = Image.open(image_list[i])   # 打开测试的图片
        plt.figure("Image")               # 图像窗口名称
        plt.imshow(img)
        plt.title(image_list[i] + '\npredict is :' + str(predict_list[i]))  # 图像题目
        plt.show()

    '''判断测试的准确率为多少'''
    correct_num = 0
    for i in range(len(image_list)):
        label = label_list[i]
        predict = predict_list[i]
        if label == predict:
            correct_num += 1
    total_num = len(image_list)
    accuary = (correct_num / total_num) * 100
    print('Test predict accuary is :{}%'.format(accuary))

if __name__ == '__main__':
    test()


