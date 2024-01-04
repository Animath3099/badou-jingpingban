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
规律：
    数据共分五类
    x为5维向量 其第n位的数字最大则分类为n

"""


class TorchModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(TorchModel, self).__init__()
        # 线性层1 将输入的数据拓展为32 * 1
        self.linear1 = nn.Linear(input_size, 32)
        # 线性层2 整合输出为分类结果 维度和待分类数量一致
        self.linear2 = nn.Linear(32, num_classes)
        self.activation = torch.softmax # 多分类使用softmax函数
        self.loss = nn.functional.cross_entropy # loss函数使用交叉熵

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        x = self.linear1(x)  # (batch_size, input_size) -> (batch_size, 32)
        x = self.linear2(x)  # (batch_size, 16) -> (batch_size, num_classes)

        #softmax dim=-1含义：?
        y_pred = self.activation(x, dim=-1)  # (batch_size, num_classes) -> (batch_size, num_classes)
        if y is not None:
            return self.loss(y_pred, y)  # 预测值和真实值计算损失
        else:
            return y_pred  # 输出预测结果





# 生成一个样本, 样本的生成方法，代表了我们要学习的规律
# 随机生成一个num_classes维向量
def build_sample():
    x = np.random.random(5)
    max_x = max(x)
    max_index = []
    for i in range(len(x)):
        if max_x == x[i]:
            max_index.append(1)
        else:
            max_index.append(0)
    return x, max_index

# 随机生成一批样本
# 正负样本均匀生成
def build_dataset(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append(y)
    return torch.FloatTensor(X), torch.FloatTensor(Y)



# 测试代码
# 用来测试每轮模型的准确率
def evaluate(model):
    model.eval()
    test_sample_num = 100
    x, y = build_dataset(test_sample_num)
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)  # 模型预测
        probability, predict = torch.max(y_pred.data, dim=1)  # 最大概率值&下标
        for y_p, y_t in zip(predict, y):  # 与真实标签进行对比
            y_true = 0
            for j in range(len(y_t)):
                if y_t[j] == 1:
                    y_true = j
                    break
            if int(y_p) == y_true:
                correct += 1 # 预测结果正确
            else:
                wrong += 1
    print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)


def main():
    # 配置参数
    epoch_num = 200  # 训练轮数
    batch_size = 200  # 每次训练样本个数
    train_sample = 500  # 每轮训练总共训练的样本总数
    input_size = 5  # 输入向量维度
    num_classes = 5  # 样本类别数量
    learning_rate = 0.001  # 学习率

    # 建立模型
    model = TorchModel(input_size, num_classes)
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
        print("epoch = ",epoch)
        print("loss = ", np.mean(watch_loss))
        acc = evaluate(model)  # 测试本轮模型结果
        log.append([acc, float(np.mean(watch_loss))])
    # 保存模型
    torch.save(model.state_dict(), "model.pt")
    # 画图
    # print(log)
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    plt.legend()
    plt.show()
    return


# 使用训练好的模型做预测
def predict(model_path, input_vec):
    input_size = 5
    num_classes = 5
    model = TorchModel(input_size, num_classes)
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重
    # print(model.state_dict())

    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        result = model.forward(torch.FloatTensor(input_vec))  # 模型预测
    for vec, res in zip(input_vec, result):
        probability, predict = torch.max(res.data, dim=0)
        print("test:")
        # print("预测类别：%d, 概率值：%f" % (predict, probability))  # 打印结果
        print("输入：%s, 预测类别：%d, 概率值：%f" % (vec, predict, probability))  # 打印结果


if __name__ == "__main__":
    main()
    test_vec = [[0.07889086,0.15229675,0.31082123,0.03504317,0.18920843],
                [0.94963533,0.5524256,0.95758807,0.95520434,0.84890681],
                [0.78797868,0.67482528,0.13625847,0.34675372,0.19871392],
                [0.19349776,0.59416669,0.92579291,0.41567412,0.7358894]]
    predict("model.pt", test_vec)
