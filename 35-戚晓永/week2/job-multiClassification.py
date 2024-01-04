# coding:utf8

import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt

"""
【第二周作业】修改torchdemo中的任务为多分类任务，完成训练。

任务描述:
1、准备一个标注数据集生成的方法，可以生成指定数量的数据集。这个数据集中的每条数据为一个n维向量，数据集共有n个类别，一个向量m维度上的数最大，则one_hot向量上的对应维度为1
2、创建一个模型:
    1)支持分类个数、学习率自定义
    2)前向计算方式为：输入层(n维)--》线性层（n维->n维）--》激活函数(sigmod)--》输出层(n维)，
    3)使用交叉熵函数计算loss；
    4)优化模型使用adma
    
"""


def build_dataset(account, dims=5):
    X = []
    Y = []
    # 创建dims维的单位矩阵
    identity_matrix = np.eye(dims)
    for i in range(account):
        x, y = build_sample(dims)
        X.append(x)
        # 转为one_hot向量
        Y.append(identity_matrix[y])
    return torch.FloatTensor(X), torch.FloatTensor(Y)


def build_sample(dims):
    x = np.random.random(dims)
    # 找到最大元素的索引
    return x, np.argmax(x)


# 2、创建一个模型，其网络模型为 输入层(n维)--》线性层（n维->n维）--》激活函数(sigmod，n维)--》输出层(n维)，使用交叉熵函数计算loss
class MultiClassficationModel(nn.Module):
    def __init__(self, dims, lr=0.01):
        super().__init__()
        self.linear = nn.Linear(dims, dims)  # 线性层
        self.activation = torch.sigmoid  # sigmoid归一化函数
        self.loss = nn.functional.cross_entropy  # loss函数采用交叉熵
        self.optimizer = torch.optim.Adam(self.parameters(), lr)  # 定义优化模型，设置学习率

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        y_pred = self.linear(x)
        y_pred = self.activation(y_pred)
        if y is not None:
            return self.loss(y_pred, y)  # 预测值和真实值计算损失
        else:
            return y_pred  # 输出预测结果


def main():
    # 配置参数
    num_classes = 5
    epoch_num = 20  # 训练轮数
    batch_size = 20  # 每次训练样本个数
    sample_account = 5000  # 每轮训练总共训练的样本总数
    learning_rate = 0.1  # 学习率
    # 建立模型
    model = MultiClassficationModel(num_classes, learning_rate)
    # 创建训练集
    X, Y = build_dataset(sample_account, num_classes)
    loss_history = []
    for epoch in range(epoch_num):
        last_loss = 0
        for batch_index in range(sample_account // batch_size):
            X_batch = X[batch_index * batch_size:(batch_index + 1) * batch_size]
            Y_batch = Y[batch_index * batch_size:(batch_index + 1) * batch_size]

            loss = model(X_batch, Y_batch)  # 计算损失值
            loss.backward()  # 计算梯度
            model.optimizer.step()  # 更新参数权重
            model.optimizer.zero_grad()  # 将累计梯度归零
        else:
            # 打印当前轮的损失
            print("第%d轮迭代后的loss为:%f" % (epoch, loss))
            loss_history.append(loss.item())
    # 保存模型
    torch.save(model.state_dict(), "multiClassification.pt")

    plt.plot(range(epoch_num), loss_history, label="loss")  # 画loss曲线
    plt.legend()
    plt.show()
    return


# 使用训练好的模型做预测
def predict(model_path, input_vec, result):
    model = MultiClassficationModel(5)
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重
    print(model.state_dict())

    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        preds = model.forward(torch.FloatTensor(input_vec))  # 模型预测
    for vec, res, pred in zip(input_vec, result, preds):
        print("输入：%s, 实际类别为：%s,预测类别：%s, 预测输出向量为：%s，预测是否准确：%s" % (vec, res.item(), torch.argmax(pred).item(), pred,res.item()==torch.argmax(pred).item()))  # 打印结果


if __name__ == "__main__":
    main()
    test_vec, result = build_dataset(10, 5)
    predict("multiClassification.pt", test_vec, [ np.argmax(i) for i in result])
