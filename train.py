from getdata import DogsVSCatsDataset as DVCD
from torch.utils.data import DataLoader as DataLoader
from models.resnet_all import *
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np


dataset_dir = './flowers'             # 数据集路径
model_cp = './checkpoint'               # 网络参数保存位置
workers = 10                        # PyTorch读取数据线程数量
batch_size = 16                     # batch_size大小
lr = 0.0001                         # 学习率
epochs = 10


def train():
    datafile = DVCD('train', dataset_dir)                                                           # 实例化一个数据集
    dataloader = DataLoader(datafile, batch_size=batch_size, shuffle=True, num_workers=workers)     # 用PyTorch的DataLoader类封装，实现数据集顺序打乱，多线程读取，一次取多个数据等效果

    print('Dataset loaded! length of train set is {0}'.format(len(datafile)))

    model = resnet152(pretrained=True)    # 实例化一个网络
    # 提取fc层中固定的参数
    fc_features = model.fc.in_features
    # 修改类别为5
    model.fc = nn.Linear(fc_features, 5)
    '''总共有5个类别的花，最后的fc层需要改成5'''

    model = model.cuda()                # 网络送入GPU，即采用GPU计算，如果没有GPU加速，可以去掉".cuda()"
    model.train()                       # 网络设定为训练模式，有两种模式可选，.train()和.eval()，训练模式和评估模式，区别就是训练模式采用了dropout策略，可以放置网络过拟合

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)         # 实例化一个优化器，即调整网络参数，优化方式为adam方法

    criterion = torch.nn.CrossEntropyLoss()                         # 定义loss计算方法，cross entropy，交叉熵，可以理解为两者数值越接近其值越小

    cnt = 0             # 训练图片数量
    for epoch in range(epochs):
        tq = tqdm(total=4323)
        tq.set_description('epoch %d, lr %f' % (epoch, lr))
        loss_record = []

                                                                        # 读取数据集中数据进行训练，因为dataloader的batch_size设置为16，所以每次读取的数据量为16，即img包含了16个图像，label有16个
        for img, label in dataloader:                                   # 循环读取封装后的数据集，其实就是调用了数据集中的__getitem__()方法，只是返回数据格式进行了一次封装
            img, label = img.cuda(), label.cuda()                       # 将数据放置在PyTorch的Variable节点中，并送入GPU中作为网络计算起点
                                                                        # input==torch.Size([16, 3, 200, 200])   output==torch.Size([16, 2])
            out = model(img)                                            # 计算网络输出值，就是输入网络一个图像数据，输出猫和狗的概率，调用了网络中的forward()方法
            loss = criterion(out, label.squeeze())      # 计算损失，也就是网络输出值和实际label的差异，显然差异越小说明网络拟合效果越好，此处需要注意的是第二个参数，必须是一个1维Tensor
            tq.update(batch_size)
            tq.set_postfix(loss='%.6f' % loss)
            loss.backward()                             # 误差反向传播，采用求导的方式，计算网络中每个节点参数的梯度，显然梯度越大说明参数设置不合理，需要调整
            optimizer.step()                            # 优化采用设定的优化方法对网络中的各个参数进行调整
            optimizer.zero_grad()                       # 清除优化器中的梯度以便下一次计算，因为优化器默认会保留，不清除的话，每次计算梯度都回累加
            cnt += 1
            loss_record.append(loss.item())
        tq.close()
        loss_train_mean = np.mean(loss_record)
        print('loss is: ', loss_train_mean)
    torch.save(model.state_dict(), '{0}/model.pth'.format(model_cp))            # 训练所有数据后，保存网络的参数


if __name__ == '__main__':
    train()










