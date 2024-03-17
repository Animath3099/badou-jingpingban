# coding:utf8

import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt

"""

基于pytorch框架编写模型训练
实现一个自行构造的找规律(机器学习)任务
规律：x是一个5维向量，如果第1个数>第5个数，则为正样本，反之为负样本

"""
# device = torch.device( "cuda:0" )


class TorchModel(nn.Module):
    def __init__(self, input_size):
        super(TorchModel, self).__init__()
        self.linear = nn.Linear(input_size, 5)  # 线性层
        self.activation = torch.sigmoid  # sigmoid归一化函数
        # self.loss = nn.functional.cross_entropy  # loss函数采用均方差损失

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        x = self.linear(x)  # (batch_size, input_size) -> (batch_size, 1)
        # x = x.to(device)
        y_pred = self.activation(x)  # (batch_size, 1) -> (batch_size, 1)
        # y_pred = y_pred.to(device)

        return y_pred  # 给出样本x，根据神经网络算出来的y_pred

        # target = torch.LongTensor([0,1,2,3,4])
        # max_index = torch.max(target)
        # print(max_index)
        # if y is not None:
        #     return self.loss(x, y)  # 预测值和真实值计算损失
        # else:
        #     return y_pred  # 输出预测结果


# 生成一个样本, 样本的生成方法，代表了我们要学习的规律
# 随机生成一个5维向量，如果第一个值最大，认为是第一类，一共五类
def build_sample():
    x = np.random.random(5)
    if x[0] == max(x):
        return x, 0
    elif x[1] == max(x):
        return x, 1
    elif x[2] == max(x):
        return x, 2
    elif x[3] == max(x):
        return x, 3
    elif x[4] == max(x):
        return x, 4

# 随机生成一批样本
# 正负样本均匀生成
def build_dataset(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append([y])
    # print(X)
    # print(Y)
    return torch.FloatTensor(X), torch.LongTensor(Y)


# 测试代码
# 用来测试每轮模型的准确率
def evaluate(model):
    model.eval()
    test_sample_num = 100
    x, y = build_dataset(test_sample_num)

    # print(y)
    count_0 = (y == 0).sum().item()
    count_1 = (y == 1).sum().item()
    count_2 = (y == 2).sum().item()
    count_3 = (y == 3).sum().item()
    count_4 = (y == 4).sum().item()

    print("本次预测集中共有%d个一类，%d个二类，%d个三类，%d个四类，%d个五类" % (count_0, count_1, count_2, count_3, count_4))
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)  # 模型预测
        # print(y_pred)
        # print(y)
        for y_p, y_t in zip(y_pred, y):  # 与真实标签进行对比
            # print(y_p)
            # print(y_t)
            if torch.argmax(y_p) == y_t:
                correct += 1  # 样本判断正确
            # elif float(y_p) < 0.4 and float(y_p) >= 0.2 and int(y_t) == 1:
            #     correct += 1  # 正样本判断正确
            # elif float(y_p) < 0.6 and float(y_p) >= 0.4 and int(y_t) == 2:
            #     correct += 1  # 正样本判断正确
            # elif float(y_p) < 0.8 and float(y_p) >= 0.6 and int(y_t) == 3:
            #     correct += 1  # 正样本判断正确
            # elif float(y_p) <= 1 and float(y_p) >= 0.8 and int(y_t) == 4:
            #     correct += 1  # 正样本判断正确
            else:
                wrong += 1
    print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)


def main():
    # 配置参数
    epoch_num = 20  # 训练轮数
    batch_size = 5  # 每次训练样本个数
    train_sample = 5000  # 每轮训练总共训练的样本总数
    input_size = 5  # 输入向量维度
    learning_rate = 0.001  # 学习率
    # 建立模型
    model = TorchModel(input_size)
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    # 创建训练集，正常任务是读取训练集
    train_x, train_y = build_dataset(train_sample)
    criterion = nn.CrossEntropyLoss()
    # 训练过程
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch_index in range(train_sample // batch_size):    
            x = train_x[batch_index * batch_size : (batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size : (batch_index + 1) * batch_size]
            y=torch.squeeze(y)
            # print(x)
            # print(y)
            y_pred = model(x)
            # print(y_pred)
            # print(y.shape)
            loss = criterion(y_pred, y)  # 计算loss
            # print('loss是什么？', loss)
            loss.backward()  # 计算梯度
            optim.step()  # 更新权重
            optim.zero_grad()  # 梯度归零
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model)  # 测试本轮模型结果
        log.append([acc, float(np.mean(watch_loss))])
    # 保存模型
    torch.save(model.state_dict(), "model.pt")
    # 画图
    print(log)
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    plt.legend()
    plt.show()
    return


# 使用训练好的模型做预测
def predict(model_path, input_vec):
    input_size = 5
    model = TorchModel(input_size)
    # model = model.to(device)
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重
    print(model.state_dict())

    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        result = model.forward(torch.FloatTensor(input_vec))  # 模型预测
        # print(result)
    for vec, res in zip(input_vec, result):
        # print('vec和res分别是', vec, res)
        print("输入：%s, 预测类别：第%s类" % (vec, torch.argmax(res).item()+1))  # 打印结果


if __name__ == "__main__":
    main()
    test_vec = [[0.07889086,0.15229675,0.31082123,0.03504317,0.18920843],
                [0.94963533,0.5524256,0.95758807,0.95520434,0.84890681],
                [0.78797868,0.67482528,0.13625847,0.34675372,0.19871392],
                [0.19349776,0.59416669,0.92579291,0.41567412,0.7358894]]
    # test_vec = test_vec.to(device)
    predict("model.pt", test_vec)
