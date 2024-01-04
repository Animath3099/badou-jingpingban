# coding:utf8

import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt


"""
规律：给定一个8维向量 x，根据以下条件将其分为三个类别：

如果第1个数 > 第3个数 且 第5个数 > 第7个数，则为类别1。
如果第3个数 > 第6个数 且 第2个数 > 第4个数，则为类别2。
否则，为类别3。

[0.8, 0.4, 0.6, 0.2, 0.9, 0.3, 0.7, 0.1] 类别1
[0.2, 0.7, 0.5, 0.9, 0.1, 0.8, 0.4, 0.6] 类别2
[0.6, 0.2, 0.8, 0.4, 0.7, 0.1, 0.5, 0.3] 类别3
"""


class MyTorchModel(nn.Module):
    def __init__(self,input_size,hidden_size,num_classes):
        super(MyTorchModel, self).__init__()
        self.layer1 = nn.Linear(input_size,hidden_size)
        self.relu = nn.ReLU() # ReLU激活函数
        self.layer2 = nn.Linear(hidden_size,num_classes)
        self.loss = nn.CrossEntropyLoss()#交叉熵损失函数(包含softmax函数)

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self,x,y=None):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        if y is not None:
            loss = self.loss(x, y)
            return loss
        else:
            return x
#如果第1个数 > 第3个数 且 第5个数 > 第7个数，则为类别1。
#如果第3个数 > 第6个数 且 第2个数 > 第4个数，则为类别2。
def build_simple():
    x = np.random.random(8)
    if x[0] > x[2] and x[4] > x[6]:
        return x, 0
    elif x[2] > x[5] and x[1] > x[3]:
        return x, 1
    else:
        return x, 2

# for _ in range(10):  # 输出10个样本
#     x, y = build_simple()
#     print(f'输入向量: {x}, 类别: {y}')
#建造数据集
def build_dataset(total_sample_num):
    X=[]
    Y=[]
    for i in range(total_sample_num):
        x,y = build_simple()
        X.append(x)
        Y.append(y)
    #x是浮点数 用FloatTensor，Y 是整数标签用LongTensor
    return torch.FloatTensor(X), torch.LongTensor(Y)

def evaluate(model):
    model.eval()
    test_sample_num = 100
    x,y = build_dataset(test_sample_num)
    print("本次预测集中共有%d个类别1，%d个类别2，%d个类别3" % (sum(y==0),  sum(y == 1), sum(y == 2)))
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)
        # print("预测结果的形状:", y_pred.shape)
        predicted_labels = torch.argmax(y_pred, dim=1)
        for pred_label, true_label in zip(predicted_labels, y):  # 与真实标签进行对比
            if pred_label == true_label:
                correct += 1
            else:
                wrong += 1

        accuracy = correct / (correct + wrong)
        print("正确预测个数：%d, 正确率：%f" % (correct, accuracy))
        return accuracy


def main():
    # 配置参数
    epoch_num = 50  # 训练轮数
    batch_size = 10 # 每次训练样本个数
    train_sample = 10000  # 每轮训练总共训练的样本总数
    input_size = 8  # 输入向量维度
    hidden_size = 30 #设置隐藏层
    num_classes = 3  #输出三类别
    learning_rate = 0.001 #学习率
    # 建立模型
    model = MyTorchModel(input_size,hidden_size,num_classes)
    #选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    # 创建训练集
    train_x, train_y = build_dataset(train_sample)
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
        acc = evaluate(model)
        log.append([acc, float(np.mean(watch_loss))])

    torch.save(model.state_dict(), "Mymodel.pt")
    # 画图
    print(log)
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    plt.legend()
    plt.show()
    return


# 使用训练好的模型做预测
def predict(model_path, input_vec):
    input_size = 8
    hidden_size = 30
    num_classes = 3
    model = MyTorchModel(input_size, hidden_size, num_classes)
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重
    print(model.state_dict())
    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        result = model.forward(torch.FloatTensor(input_vec))  # 模型预测
        predicted_labels = torch.argmax(result, dim=1)  # 获取概率最大的类别
    for vec, pred_label in zip(input_vec, predicted_labels):
        print(f"输入：{vec}, 预测类别：{pred_label.item()}")
if __name__ == "__main__":
    main()
    test_vec = np.random.rand(10, 8)
    predict("Mymodel.pt", test_vec)