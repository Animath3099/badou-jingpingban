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

修改为多分类
随机生成一个5维向量，如果第一个值大于第五个值，认为是0样本，如果第二个值大于第四个值视为1样本，其他视为2样本
采用交叉熵损失函数
"""


class TorchModel(nn.Module):
    def __init__(self, input_size):
        super(TorchModel, self).__init__()
        # self.linear = nn.Linear(input_size, 1)  # 线性层
        self.fc = nn.Linear(input_size, input_size*64)  # 全连接层
        self.linear = nn.Linear(input_size*64, 3)  # 修改为分类数量
        self.activation = torch.sigmoid  # sigmoid归一化函数
        # self.loss = nn.functional.mse_loss  # loss函数采用均方差损
        self.loss = nn.functional.cross_entropy  # loss采用交叉熵损失

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        # x int,这个int类型要如何转变为float？

        # 正常的流程应该是x先切词再转化为词表中对应的id,这时数据类型为int
        # 之后，经过embedding层增加hs维度，数据类型转化为float类型，再送入模型
        # 模型的权重和偏执一般都定义为float类型，不能直接使用int类型
        # 这里的x是自定义的数据，它代表的应该是经过embedding后的结果，所以应该直接定义为float类型值
        x_fc = self.fc(x)
        y_pred = self.linear(x_fc)  # (batch_size, input_size) -> (batch_size, 2)
        # y_pred = self.activation(x) # 使用交叉熵就不用归一化到01之间 # 20,2 y.squeeze() [20]  # (batch_size, 1) -> (batch_size, 2)
        if y is not None:
            return self.loss(y_pred, y.squeeze())  # 预测值和真实值计算损失
        else:
            return y_pred  # 输出预测结果


# 生成一个样本, 样本的生成方法，代表了我们要学习的规律
# 随机生成一个5维向量，如果第一个值大于第五个值，认为是正样本，反之为负样本
def build_sample():
    x = np.random.random(5)
    if x[0] > x[4]:
        return x, 1
    else:
        return x, 0

# 随机生成一个5维向量，如果第一个值大于第五个值，认为是0样本，如果第二个值大于第四个值视为1样本，其他视为2样本
def build_multiclass_sample():
    x = np.random.random(5)
    if x[0] > x[4]:
        # 类别 1
        return x, 0
    elif x[1] > x[3]:
        # 类别 2
        return x, 1
    else:
        # 类别 3
        return x, 2


# 随机生成一批样本
# 正负样本均匀生成
def build_dataset(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        # x, y = build_sample()
        x, y = build_multiclass_sample()  # 生成多分类样本
        X.append(x)
        Y.append([y])
    return torch.FloatTensor(X), torch.LongTensor(Y)


# 测试代码
# 用来测试每轮模型的准确率
def evaluate(model):
    model.eval()
    test_sample_num = 100
    x, y = build_dataset(test_sample_num)
    stats_class = {}
    stats_dict = {"correct": 0, "wrong": 0}  # 用于存储测试结果
    with torch.no_grad():
        y_pred = model(x)  # 模型预测
        for y_p, y_t in zip(y_pred, y):  # 与真实标签进行对比
            pred_label = torch.argmax(y_p)
            if int(y_t) == int(pred_label):
                stats_dict["correct"] += 1
            else:
                stats_dict["wrong"] += 1
            int_label = int(y_t.data.type(torch.DoubleTensor).cpu().numpy().tolist()[0])  # 将标签转为int
            if int_label not in stats_class:
                stats_class[int_label] = 1
            else:
                stats_class[int_label] += 1
    correct = stats_dict["correct"]
    wrong = stats_dict["wrong"]
    for i in stats_class:
        print("%s样本数量：%d" % (i, stats_class[i]))
    print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)


def main():
    # 配置参数
    epoch_num = 10  # 训练轮数
    batch_size = 20  # 每次训练样本个数
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
    # 训练过程
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch_index in range(train_sample // batch_size):
            x = train_x[batch_index * batch_size: (batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size: (batch_index + 1) * batch_size]
            loss = model(x, y)  # 计算loss
            loss.backward()  # 计算梯度
            if batch_index % 2 == 0:
                optim.step()  # 更新权重
                optim.zero_grad()  # 梯度归零
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model)  # 测试本轮模型结果
        log.append([acc, float(np.mean(watch_loss))])
    # 保存模型
    torch.save(model.state_dict(), "model.pth")
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

