import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt

"""
基于pytorch框架编写模型训练
实现一个自行构造的找规律(机器学习)任务
规律：随机生成一个3维向量，如果第一个数最大为分类狗，第二个最大为猫，第三个最大为猪
"""
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


class TorchModel(nn.Module):
    def __init__(self, input_size):
        super(TorchModel, self).__init__()
        self.linear = nn.Linear(input_size, 3)  # 线性层
        self.activation = torch.sigmoid  # sigmoid归一化函数
        self.loss = nn.CrossEntropyLoss()  # loss函数采用交叉熵损失

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        x = self.linear(x)  # (batch_size, input_size) -> (batch_size, 3)
        y_pred = self.activation(x)  # (batch_size, 1) -> (batch_size, 1)
        if y is not None:
            return self.loss(y_pred, y.squeeze())  # 预测值和真实值计算损失
        else:
            return y_pred


dict_map = {'0': 'dog', '1': 'cat', '2': 'pig'}

# 生成一个样本, 样本的生成方法，代表了我们要学习的规律
# 随机生成一个3维向量，如果第一个数最大为分类狗，第二个最大为猫，第三个最大为猪


def build_sample():
    x = np.random.random(3)
    if max(x) == x[0]:
        return x, 0
    elif max(x) == x[1]:
        return x, 1
    else:
        return x, 2


# 生成数据集
def build_dataset(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append([y])
    return torch.FloatTensor(X), torch.LongTensor(Y)


# 测试代码
# 用来测试每轮模型的准确率
def evaluate(model):
    model.eval()
    test_sample_num = 200
    x, y = build_dataset(test_sample_num)
    y = y.squeeze()
    print("本次预测集中共有%d个dog样本，%d个cat样本, %d个pig样本" % (y.tolist().count(0), y.tolist().count(1), y.tolist().count(2)))
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)
        for y_p, y_t in zip(y_pred, y):
            if int(torch.argmax(y_p)) == int(y_t):
                correct += 1
            else:
                wrong += 1
    print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)


# 训练逻辑
def main():
    # 超参数设置
    epoch_num = 20  # 训练轮数
    train_sample_num = 5000  # 训练样本总数
    batch_size = 20  # 每轮训练批量训练大小
    input_size = 3  # 向量输入维度
    learn_rate = 0.005  # 学习率
    # 建立模型
    model = TorchModel(input_size)
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learn_rate)
    log = []
    # 创建训练集
    train_x, train_y = build_dataset(train_sample_num)
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch_index in range(train_sample_num // batch_size):
            x = train_x[batch_index * batch_size: (batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size: (batch_index + 1) * batch_size]
            # 计算loss
            loss = model(x, y)
            # 计算梯度
            loss.backward()
            # 更新权重
            optim.step()
            # 梯度归零
            optim.zero_grad()
            watch_loss.append(loss.item())

        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model)
        log.append([acc, float(np.mean(watch_loss))])
    # 保存模型
    torch.save(model.state_dict(), "model_1.pt")

    # 画图
    print(log)
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    # for l in log:
    #     plt.plot(range(len(log)), l[0])
    #     plt.plot(range(len(log)), l[1])
    plt.legend()
    plt.show()
    return


def predict(model_path, input_vec):
    input_size = 3
    model = TorchModel(input_size)
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重
    print(model.state_dict())
    model.eval()  # 测试模式
    with torch.no_grad():
        result = model(torch.FloatTensor(input_vec))
        for i, vec in enumerate(input_vec):
            print(int(torch.argmax(result[i])), dict_map[str(int(torch.argmax(result[i])))], vec, result[i])  # 打印结果


if __name__ == "__main__":
    main()
    test_vec = [[0.17889086, 0.55229675, 0.31082123],
                [0.94963533, 0.5524256, 0.95758807],
                [0.78797868, 0.67482528, 0.13625847],
                [0.19349776, 0.59416669, 0.92579291],
                [0.09349776, 0.29416669, 0.32579291]
                ]

    predict("model_1.pt", test_vec)
