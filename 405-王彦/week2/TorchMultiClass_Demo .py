# coding:utf8

import torch
import torch.nn as nn
import os ##导入原因是提示有  OPENMP访问线程冲突，libiomp5md.dll， but found libiomp5md.dll already initialized
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
import numpy as np
import random
import json
import matplotlib.pyplot as plt

"""

基于pytorch框架编写模型训练
实现一个自行构造的找规律(机器学习)任务
规律：x是一个5维向量，5个向量里面最大的数所在位置即为分类的对应种类，比如在第三个位置则分类索引为2
在第一个位置则分类索引为0，以此类推
"""


class TorchMultiClassficationModel(nn.Module):
    def __init__(self, input_size):
        super(TorchMultiClassficationModel, self).__init__()
        self.linear = nn.Linear(input_size, 5)  # 线性层
        ##self.activation = torch.softmax(self.linear,0)  # softmax激活函数
        self.loss = nn.functional.cross_entropy  # loss函数采用损失熵

    # 当输入真实分类数据，返回loss值；无真实分类数据，返回预测值
    def forward(self, x, y=None):
        ##x = self.linear(x)  # (batch_size, input_size) -> (batch_size, 1)
        ##y_pred = self.activation(x)  # (batch_size, 1) -> (batch_size, 1)
        y_pred = self.linear(x)
        if y is not None:
            return self.loss(y_pred, y)  # 预测值和真实值计算损失
        else:
            return y_pred  # 输出预测结果


# 生成一个样本, 样本的生成方法，代表了我们要学习的规律
# 随机生成一个5维向量，最大值所在的位置即为分类类别，
def build_sample():
    ##x = torch.randn(5)
    x = np.random.random(5)
    maxpos = np.argmax(x)
##    print(f"random x_tensor:{x},maxpos:{maxpos}")
    if maxpos == 0:
        return x, 0
    elif maxpos == 1:
        return x, 1
    elif maxpos == 2:
        return x, 2
    elif maxpos == 3:
        return x, 3
    elif maxpos == 4:
        return x, 4

# 随机生成一批样本
def build_dataset(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append(y)
    return torch.FloatTensor(X), torch.LongTensor(Y)
    ##return X,Y

# 测试代码
# 用来测试每轮模型的准确率
def evaluate(model):
    model.eval()
    test_sample_num = 100
    x, y = build_dataset(test_sample_num)
    ##print("本次预测集中共有%d个正样本，%d个负样本" % (sum(y), test_sample_num - sum(y)))
    print(f"本次测试集， x {x},y {y}")
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)  # 模型预测
        for y_p, y_t in zip(y_pred, y):  # 与真实标签进行对比
            if torch.argmax(y_p) == int(y_t):
                correct += 1  # 负样本判断正确
            else:
                wrong += 1  # 正样本判断正确
    print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)


def main():
    # 配置参数
    epoch_num = 20  # 训练轮数
    batch_size = 20  # 每次训练样本个数
    train_sample = 5000  # 每轮训练总共训练的样本总数
    input_size = 5  # 输入向量维度
    learning_rate = 0.0001  # 学习率
    # 建立模型
    model = TorchMultiClassficationModel(input_size)
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    # 创建训练集，正常任务是读取训练集
    train_x, train_y = build_dataset(train_sample)
    print(f"本次训练集 train_x:{train_x}, train_y:{train_y}")
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
    torch.save(model.state_dict(), "model_classification.pt")
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
    model = TorchMultiClassficationModel(input_size)
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重
    print(model.state_dict())

    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        result = model.forward(torch.FloatTensor(input_vec))  # 模型预测
    for vec, res in zip(input_vec, result):
        print("输入：%s, 预测类别：%s, 概率值：%s" % (vec, torch.argmax(res), res))  # 打印结果


if __name__ == "__main__":
##    main()
    test_vec = [[0.07889086,0.15229675,0.31082123,0.03504317,0.18920843],
                [0.94963533,0.5524256,0.95758807,0.95520434,0.84890681],
                [0.78797868,0.67482528,0.13625847,0.34675372,0.19871392],
                [0.19349776,0.59416669,0.92579291,0.41567412,0.7358894],
                [0.34564678,0.564556789,0.45678394,0.45673849,0.7463527]]
    predict("model_classification.pt", test_vec)
