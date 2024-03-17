import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

class TorchModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(TorchModel, self).__init__()
        self.linear = nn.Linear(input_size, out_features=num_classes)
        # self.activation = torch.sigmoid
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, y=None):
        x = self.linear(x)
        y_pred = torch.softmax(x, dim=1)
        if y is not None:
            return self.loss(y_pred, y)  # 不需要额外的softmax
        else:
            return y_pred  # 在预测时应用softmax

def build_sample():
    x = np.random.random(3)
    y = np.argmax(x)  # 类别为最大值的索引
    return x, y
def build_dataset(total_sample_num):
    X, Y = [], []
    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append(y)
    X = np.array(X, dtype=np.float32)  # 转换为单一的Numpy数组
    Y = np.array(Y, dtype=np.int64)
    return torch.from_numpy(X), torch.from_numpy(Y)  # 使用torch.from_numpy转换

def evaluate(model):
    model.eval() #将模型设置为评估模式。这对于某些类型的层（如Dropout和BatchNorm层）很重要，因为它们在训练和评估时的行为是不同的。在评估模式下，这些层的行为会调整以便于模型能够进行准确的预测。
    #创建检测集
    test_sample_num = 100
    x, y = build_dataset(test_sample_num)
    correct = 0
    total = 0
    with torch.no_grad(): #上下文管理器=>用于暂时禁用梯度计算，节省运算时间。在评估模型时，通常不需要计算梯度，因为不会进行反向传播或优化。
        y_pred = model(x)  # 模型预测

        _, predicted = torch.max(y_pred, 1)
        # 使用 torch.max 函数获取预测结果中概率最高的类别的索引。
        # torch.max 返回两个值：最大值和它们的索引。

        total += y.size(0)
        # 将当前批次的样本数量添加到总样本数量。
        # y.size(0) 返回标签张量的第一个维度的大小，即样本数量。

        # 比较预测和真实标签
        correct += (predicted == y).sum().item()
        # 计算预测正确的样本数量。
        # (predicted == y) 比较预测的类别和真实标签是否相等，返回一个布尔张量。
        # .sum() 对这个布尔张量求和，得到正确预测的数量。
        # .item() 将这个值转换为Python标量。

        accuracy = correct / total
    print("正确预测个数：%d, 正确率：%f" % (correct, accuracy))
    return accuracy

def main():
    epoch_num = 20
    batch_size = 20
    train_sample = 5000
    input_size = 3
    learning_rate = 0.001

    model = TorchModel(input_size, 3) # 模型初始化
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate) # 优化器初始化
    log = []
    train_x, train_y = build_dataset(train_sample) # 生成训练集
    for epoch in range(epoch_num):
        model.train()  # 将模型设置为训练模式
        watch_loss = [] # 用于记录每个批次的损失
        for batch_index in range(train_sample // batch_size):
            x = train_x[batch_index * batch_size : (batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size: (batch_index + 1) * batch_size]
            loss = model(x, y)
            loss.backward()
            optim.step()
            optim.zero_grad()
            watch_loss.append((loss.item()))
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model)  # 测试本轮模型结果
        log.append([acc, float(np.mean(watch_loss))])
    # 保存模型
    torch.save(model.state_dict(), "model.pt")
    # 画图
    print(log)
    # plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
    # plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    # plt.legend()
    # plt.show()
    return
# 使用训练好的模型做预测
def predict(model_path, input_vec):
    input_size = 3
    model = TorchModel(input_size, 3)
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重
    print(model.state_dict())

    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        result = model.forward(torch.FloatTensor(input_vec))  # 模型预测
        _, predicted = torch.max(result, 1)
    for vec, pred, res in zip(input_vec, predicted, result):
        print("输入：%s, 预测类别：%d, 概率值：%f" % (vec, pred.item(), res[pred].item()))  # 打印结果

if __name__ == "__main__":
    main()
    print('模型训练结束')
    test_vec = [[0.07889086,0.15229675,0.31082123],
                [0.94963533,0.95520434,0.84890681],
                [0.78797868,0.34675372,0.19871392],
                [0.19349776,0.59416669,0.92579291]]
    predict("model.pt", test_vec)


