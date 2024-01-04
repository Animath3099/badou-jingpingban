# coding:utf8

import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from torchvision import datasets, transforms
from mnist import load_mnist
"""

基于pytorch框架编写模型训练
MNIST数据集分类

"""





# 加载训练集 测试集

(X_train1, y_train1), (X_test1, y_test1) = load_mnist() # 初始化mnist数据集 并以numpy的形式加载到内存中
# 转换为张量
X_train = torch.from_numpy(X_train1)
X_test = torch.from_numpy(X_test1)
y_train = torch.from_numpy(y_train1)
y_test = torch.from_numpy(y_test1)

X_train = X_train.float()
X_test = X_test.float()
y_train = y_train.float()
y_test = y_test.float()
# print("y_train size:", y_train.size())
X_train = X_train.reshape(60000, -1)
# y_train = y_train.reshape(60000)
X_test = X_test.reshape(10000, -1)

train_data = TensorDataset(X_train, y_train) # 创建训练数据集
test_dataset = TensorDataset(X_test, y_test) # 创建一个数据集 评价时用


# print("X_train=", X_train.size())

class TorchModel(nn.Module):
    # 多分类任务需要在定义模型的时候给出分类的数量 增加一个参数class_num
    def __init__(self, input_size, class_num):
        # super是内建函数 通常被用于在子类中调用父类的初始化方法 以便继承父类的属性和方法
        super(TorchModel, self).__init__()
        # 定义第一个全连接层
        # input_size:输入的神经元个数/输入特征的维度/输入数据的特征数量 128:
        self.linear1 = nn.Linear(input_size, 32)
        self.linear2 = nn.Linear(32, 64)
        self.linear3 = nn.Linear(64, 128)
        self.linear4 = nn.Linear(128, 64)
        # 定义第二个全连接层 输入维度和上一个全连接层的输出相同 输出维度是我们的分类类别数
        self.linear5 = nn.Linear(64, class_num)
        # 在答案中也可以只设置一个线性层 直接设置输出维度为class_num即可
        """
        # self.linear = nn.Linear(input_size, 1)
        # 二分类中只用到了一个线性层 输出的结果是0-1间的一个概率值 我们认为<0.5为0 反之为1
        """
        self.activation = torch.softmax  # 多分类使用softmax函数
        # 使用交叉熵作为多分类任务的损失函数
        self.loss = nn.functional.cross_entropy
        # 答案中取消了activation的定义 二分类中sigmoid函数是在全连接层的输出后添加
        '''
        在PyTorch中，Sigmoid函数是一种常用的激活函数，它可以将任何实数映射到 (0,1) 区间。
        在二分类问题中，Sigmoid函数的输出可以被看作是样本属于正类的概率。
        例如，如果Sigmoid函数的输出为0.7，为样本有70%的概率属于正类，相应地，有30%的概率属于负类。
        Sigmoid函数在深度较大的神经网络中可能会导致梯度消失问题 --> 选择使用ReLU函数或其变种
        多分类中将不使用sigmoid激活函数和均方差损失函数
        self.activation = torch.sigmoid  # sigmoid归一化函数
        self.loss = nn.functional.mse_loss  # loss函数采用均方差损失
        '''

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        # softmax函数 对原始数据直接使用softmax反而会使得性能下降
        # x = self.activation(x, dim=-1)
        x = self.linear1(x)  # (batch_size, input_size) -> (batch_size, 128)
        x = self.linear2(x)  # (batch_size, 128) -> (batch_size, 4)
        x = self.linear3(x)
        x = self.linear4(x)
        x = self.linear5(x)
        y_pred = self.activation(x, dim=-1) # softmax函数 dim=-1会对最后输入的维度进行操作
        # y_pred = self.activation(x)  # (batch_size, 1) -> (batch_size, 1)
        if y is not None:
            y = y.long() # 交叉熵loss函数期望目标张量为long类型
            return self.loss(y_pred, y)  # 预测值和真实值计算损失
        else:
            return y_pred  # 输出预测结果


# 评估代码
# 用来测试每轮模型的准确率 随机在训练集中抽取200个
def evaluate(model):
    model.eval()
    print("随机抽取训练集中200个样本")
    sample_size = 200 # 每次抽取200组
    #每次随机抽取20个 shuffle=True每次会随机打乱抽取
    test_loader = DataLoader(test_dataset, batch_size=sample_size, shuffle=True)
    # 从DataLoader中获取一个batch的数据
    X_sample, y_sample = next(iter(test_loader))
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(X_sample)  # 模型预测
        for y_p, y_t in zip(y_pred, y_sample):  # 与真实标签进行对比
            if torch.argmax(y_p) == int(y_t):
                correct += 1
            else:
                wrong += 1
    print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)


def main():
    # 配置参数
    epoch_num = 200  # 训练轮数
    batch_size = 100  # 每次训练样本个数
    train_sample = 500  # 每轮训练总共训练的样本总数
    # 在将每轮训练的样本总数从25增加到500之后 batch_size的修改便不会导致loss值为nan了
    # 结论--> train_sample // batch_size 若train_sample太小就不能训练了 所以loss一直是nan
    input_size = 784  # 输入向量维度 输入为图片格式28*28 拉伸为一维784
    learning_rate = 0.001  # 学习率
    class_num = 10
    print("训练集总大小:", len(X_train))
    # 建立模型
    model = TorchModel(input_size, class_num)
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []

    # 训练过程

    # 每次随机抽取batch_size个 shuffle=True每次会随机打乱抽取
    # 训练时定义的DataLoader应该是放在两重循环外而不是循环内部。
    test_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch_index in range(train_sample // batch_size):

            # 从DataLoader中获取一个batch的数据
            x, y = next(iter(test_loader))
            # print("x=", x.size())
            # print("y=", y.size())
            loss = model(x, y)  # 计算loss
            loss.backward()  # 计算梯度
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # 梯度最大值截断到1.0
            optim.step()  # 更新权重
            optim.zero_grad()  # 梯度归零
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model)  # 测试本轮模型结果
        log.append([acc, float(np.mean(watch_loss))])
    # 保存模型
    torch.save(model.state_dict(), "../model.pt")
    # 画图
    print(log)
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    plt.legend()
    plt.show()
    return


# 使用训练好的模型做预测
def predict(model_path, input_vec):
    input_size = 784
    class_num = 10
    model = TorchModel(input_size, class_num)
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重
    # print(model.state_dict())

    eval_correct = 0
    eval_wrong = 0
    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        y_pred = model.forward(input_vec)  # 模型预测
        for y_p, y_t in zip(y_pred, y_test):
            if torch.argmax(y_p) == int(y_t):
                eval_correct += 1
            else:
                eval_wrong += 1

        print("测试集的结果:")
        # print("预测类别：%d, 概率值：%f" % (predict, probability))  # 打印结果
        print("测试集大小：%d, 正确率%f" % (len(y_test),eval_correct/(eval_correct+eval_wrong)))  # 打印结果




if __name__ == "__main__":
    main()
    predict("../model.pt", X_test)
