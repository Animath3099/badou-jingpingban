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
规律：x是一个4维向量，
如果排序是正序，则为1类样本
如果排序是倒叙，则为2类样本
如果排序是乱序，则为0类样本

"""


class TorchModel(nn.Module):
    def __init__(self, input_size):
        super(TorchModel, self).__init__()
        self.linear = nn.Linear(input_size, 3)  # 线性层
        self.activation = torch.softmax  # sigmoid归一化函数
        self.loss = nn.functional.cross_entropy

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        x = self.linear(x)  # (batch_size, input_size) -> (batch_size, 3)
        y_pred = self.activation(x, 1)  # (batch_size, 3) -> (batch_size, 3)
        if y is not None:
            return self.loss(y_pred, y)  # 预测值和真实值计算损失
        else:
            return y_pred  # 输出预测结果


# 生成一个样本, 样本的生成方法，代表了我们要学习的规律
# 随机生成一个3维向量
# 如果排序是正序，则为1类样本
# 如果排序是倒叙，则为2类样本
# 如果排序是乱序，则为0类样本
def build_sample(i):
    if i % 3 == 0:
        x = [random.randint(0, 10), random.randint(11, 30), random.randint(31, 60), random.randint(61, 100)]
        y = [0, 0, 1]
    elif i % 3 == 1:
        x = [random.randint(61, 100), random.randint(31, 60), random.randint(11, 30), random.randint(0, 10)]
        y = [0, 1, 0]
    else:
        x = [random.randint(0, 100) for _ in range(4)]
        if x[3] > x[2] > x[1] > x[0]:
            y = [0, 0, 1]
        elif x[0] > x[1] > x[2] > x[3]:
            y = [0, 1, 0]
        else:
            y = [1, 0, 0]
    return x, y


# 随机生成一批样本
def build_dataset(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample(i)
        X.append(x)
        Y.append(y)
    return torch.FloatTensor(X), torch.FloatTensor(Y)


# 测试代码
# 用来测试每轮模型的准确率
def evaluate(model):
    model.eval()
    test_sample_num = 100
    x, y = build_dataset(test_sample_num)
    # print("本次预测集中共有%d个正样本，%d个负样本" % (sum(y), test_sample_num - sum(y)))
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)  # 模型预测
        for y_p, y_t in zip(y_pred.numpy().tolist(), y.numpy().tolist()):  # 与真实标签进行对比
            if float(y_p[0]) < 0.5 > float(y_p[1]) and float(y_p[2]) >= 0.5 and y_t == [0, 0, 1]:
                correct += 1
            elif float(y_p[0]) < 0.5 > float(y_p[2]) and float(y_p[1]) >= 0.5 and y_t == [0, 1, 0]:
                correct += 1
            elif float(y_p[1]) < 0.5 > float(y_p[2]) and float(y_p[0]) >= 0.5 and y_t == [1, 0, 0]:
                correct += 1
            else:
                wrong += 1
    print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)


def main():
    # 配置参数
    epoch_num = 100  # 训练轮数
    batch_size = 30  # 每次训练样本个数
    train_sample = 5000  # 每轮训练总共训练的样本总数
    input_size = 4  # 输入向量维度
    learning_rate = 0.001  # 学习率
    # 建立模型
    model = TorchModel(input_size)
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    # 创建训练集，正常任务是读取训练集
    train_x, train_y = build_dataset(train_sample)
    # 训练过程
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch_index in range(train_sample // batch_size):
            x = train_x[batch_index * batch_size: (batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size: (batch_index + 1) * batch_size]
            loss = model(x, y)  # 计算loss
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
    input_size = 4
    model = TorchModel(input_size)
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重
    print(model.state_dict())

    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        result = model.forward(torch.FloatTensor(input_vec))  # 模型预测
    for vec, res in zip(input_vec, result.numpy().tolist()):
        if float(res[0]) < 0.5 > float(res[1]) and float(res[2]) >= 0.5:
            abc = "正序"
        elif float(res[0]) < 0.5 > float(res[2]) and float(res[1]) >= 0.5:
            abc = "倒序"
        else:
            abc = "乱序"

        print("输入：%s, 预测类别：%s" % (vec, abc))  # 打印结果


if __name__ == "__main__":
    main()
    test_vec = [[12, 11, 5, 1],
                [5, 3, 16, 2],
                [1, 5, 9, 20],
                [14, 51, 77, 99],
                [98, 53, 54, 22]]
    predict("model.pt", test_vec)
