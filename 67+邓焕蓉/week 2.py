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
规律：x是一个5维向量，如果第1个数>四分之一位点，则为第一类样本，
如果第1个数>二分之一位点，则为第二类样本，
如果第1个数>四分之三位点，则为第三类样本，反之为第四类样本

"""


class TorchModel(nn.Module):
    def __init__(self, input_size):
        super(TorchModel, self).__init__()
        self.linear = nn.Linear(input_size, 1)  # 线性层
        self.activation = torch.sigmoid  # sigmoid归一化函数
        self.loss = nn.functional.mse_loss  # loss函数采用均方差损失
        # self.loss = nn.CrossEntropyLoss() # loss函数使用交叉熵计算

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        x = self.linear(x)  # (batch_size, input_size) -> (batch_size, 1)
        y_pred = self.activation(x)  # (batch_size, 1) -> (batch_size, 1)
        if y is not None:
            return self.loss(y_pred, y)  # 预测值和真实值计算损失
        else:
            return y_pred  # 输出预测结果


# 生成一个样本, 样本的生成方法，代表了我们要学习的规律
# 随机生成一个5维向量，如果第1个数>四分之一位点，则为第一类样本，
# 如果第1个数>二分之一位点，则为第二类样本，
# 如果第1个数>四分之三位点，则为第三类样本，反之为第四类样本
def build_sample():
    x = np.random.random(5)
    if x[0] > (np.max(x)-np.min(x))* 1 / 4 + np.min(x):
        return x, 0
    if x[0] > (np.max(x)-np.min(x))* 2 / 4 + np.min(x):
        return x, 1
    if x[0] > (np.max(x)-np.min(x))* 3 / 4 + np.min(x):
        return x, 2
    else:
        return x, 3


# 随机生成一批样本
# 四类样本均匀生成
def build_dataset(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append([y])
    return torch.FloatTensor(X), torch.FloatTensor(Y)


# 测试代码
# 用来测试每轮模型的准确率
def evaluate(model):
    model.eval()
    # 随机分类
    test_sample_num = 100
    test_num = 4

    def allocation_amount(test_num, test_sample_num):
        a = [np.random.randint(0, test_sample_num) for i in range(test_num - 1)]  # 生成整数随机数
        a.append(0)
        a.append(test_sample_num)
        a.sort()
        b = [a[i + 1] - a[i] for i in range(test_num)]  # 列表推导式，计算列表中每两个数之间的间隔
        b = np.array(b)
        return b

    num = allocation_amount(test_num, test_sample_num)  # 随机生成一个长度为test_num， 总和为test_sample_num的列表
    num_a = num[0]
    num_b = num[1]
    num_c = num[2]
    num_d = num[3]

    x, y = build_dataset(test_sample_num)
    print("本次预测集中共有%d个第一类样本，%d个第二类样本，%d个第三类样本，%d个第四类样本" % (num_a, num_b, num_c, num_d))
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)  # 模型预测
        for y_p, y_t in zip(y_pred, y):  # 与真实标签进行对比
            if float(y_p) < 0.5 and int(y_t) == 0:
                correct += 1  # 第一类样本判断正确
            elif 0.5 <= float(y_p) < 1.5 and int(y_t) == 1:
                correct += 1  # 第二类样本判断正确
            elif 1.5 <= float(y_p) < 2.5 and int(y_t) == 2:
                correct += 1  # 第三类样本判断正确
            elif 2.5 <= float(y_p) and int(y_t) == 3:
                correct += 1  # 第四类样本判断正确
            else:
                wrong += 1
    print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)


def main():
    # 配置参数
    epoch_num = 80  # 训练轮数
    batch_size = 20  # 每次训练样本个数
    train_sample = 5000  # 每轮训练总共训练的样本总数
    input_size = 5  # 输入向量维度
    learning_rate = 0.002  # 学习率
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
            x = train_x[batch_index * batch_size : (batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size : (batch_index + 1) * batch_size]
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
    input_size = 5
    model = TorchModel(input_size)
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重
    print(model.state_dict())

    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        result = model.forward(torch.FloatTensor(input_vec))  # 模型预测
    for vec, res in zip(input_vec, result):
        print("输入：%s, 预测类别：%d, 概率值：%f" % (vec, round(float(res)), res))  # 打印结果


if __name__ == "__main__":
    main()
    test_vec = [[0.07889086,0.15229675,0.31082123,0.03504317,0.18920843],
                [0.94963533,0.5524256,0.95758807,0.95520434,0.84890681],
                [0.78797868,0.67482528,0.13625847,0.34675372,0.19871392],
                [0.19349776,0.59416669,0.92579291,0.41567412,0.7358894]]
    predict("model.pt", test_vec)
