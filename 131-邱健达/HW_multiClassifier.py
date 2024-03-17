"""

基于torch框架编写线性层+激活层的多分类模型训练任务
任务1制定: 输入一维向量x, 返回最大元素下标
任务2制定: 输入一维向量x, 返回次大元素下标
任务3制定: 输入一维向量x, 返回元素大小在各区间的数量 < 非分类 >

"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

class MultiClassficationTorchModel(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.linear = nn.Linear(input_size, input_size)
        self.loss = nn.functional.cross_entropy # torch的ce自带softmax

    def forward(self, x, y=None):
        y_pred = self.linear(x)
        if y is not None:
            return self.loss(y_pred, y)
        else:
            return y_pred

def build_sample1():
    sample = np.random.rand(5)
    return sample, np.argmax(sample)

def build_dataset1(length):
    X, Y = [], []
    for ix in range(length):
        x, y = build_sample1()
        X.append(x)
        Y.append(y)
    return torch.FloatTensor(X), torch.LongTensor(Y)

def build_sample2():
    sample = np.random.rand(5)
    return sample, np.argmax(sample)

def build_dataset2(length):
    X, Y = [], []
    for ix in range(length):
        x, y = build_sample2()
        X.append(x)
        Y.append(y)
    return torch.FloatTensor(X), torch.LongTensor(Y)

def evaluate1(model, length):
    model.eval()
    x, y = build_dataset1(length)
    count = [0] * 5
    for elem in y:
        count[elem] += 1
    for elem in y:
        print(f"最大元素在下标0的样本数: {count[elem]}")
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)
        for y_p, y_r in zip(y_pred, y):
            if np.argmax(y_p) == int(y_r):
                correct += 1
            else:
                wrong += 1
    print(f"正确预测个数: {correct}, 正确率: {correct / ( correct + wrong )}")
    return correct / ( correct + wrong )

def evaluate2(model, length):
    model.eval()
    x, y = build_dataset2(length)
    count = [0] * 5
    for elem in y:
        count[elem] += 1
    for elem in y:
        print(f"最大元素在下标0的样本数: {count[elem]}")
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)
        for y_p, y_r in zip(y_pred, y):
            if np.argmax(y_p) == int(y_r):
                correct += 1
            else:
                wrong += 1
    print(f"正确预测个数: {correct}, 正确率: {correct / (correct + wrong)}")
    return correct / (correct + wrong)

def predict(model_path, input_vec):
    model = MultiClassficationTorchModel(5)
    model.load_state_dict(torch.load(model_path))
    print(model.state_dict)

    model.eval()
    with torch.no_grad():
        result = model(torch.FloatTensor(input_vec))
    for vec, res in zip(input_vec, result):
        print(f"输入向量: {vec}, 预测下标: {torch.argmax(res)}, 该模型下对各类型的置信水平: {res}")

def task1():
    train_size = 5000
    X_train, Y_train = build_dataset1(train_size)
    print(X_train)
    print(Y_train)
    # 超参数
    batch_size = 20
    epoch_size = 100
    lr = .001

    model = MultiClassficationTorchModel(5)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    CE = []
    for epoch in range(epoch_size):
        model.train()
        watch_loss = []
        for batch_index in range(train_size // batch_size):
            x = X_train[batch_index * batch_size : (batch_index + 1) * batch_size]
            y = Y_train[batch_index * batch_size : (batch_index + 1) * batch_size]
            loss = model(x, y)
            loss.backward()
            optim.step()
            optim.zero_grad()
            watch_loss.append(loss.item())
        print(f"第{epoch}轮的平均CE: {np.mean(watch_loss)}")
        CE.append([evaluate1(model, 100), np.mean(watch_loss)])

    torch.save(model.state_dict(), "model1.pt")

    print(CE)
    plt.plot(range(len(CE)), [ce[0] for ce in CE], label="ACC")
    plt.plot(range(len(CE)), [ce[1] for ce in CE], label="CE")
    plt.legend()
    plt.show()
    return

def task2():
    train_size = 5000
    X_train, Y_train = build_dataset1(train_size)

    # 超参数
    batch_size = 20
    epoch_size = 100
    lr = .001

    model = MultiClassficationTorchModel(5)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    CE = []
    for epoch in range(epoch_size):
        model.train()
        watch_loss = []
        for batch_index in range(train_size // batch_size):
            x = X_train[batch_index * batch_size : (batch_index + 1) * batch_size]
            y = Y_train[batch_index * batch_size : (batch_index + 1) * batch_size]
            loss = model(x, y)
            loss.backward()
            optim.step()
            optim.zero_grad()
            watch_loss.append(loss.item())
        print(f"第{epoch}轮的平均CE: {np.mean(watch_loss)}")
        CE.append([evaluate2(model, 100), np.mean(watch_loss)])

    torch.save(model.state_dict(), "model2.pt")

    print(CE)
    plt.plot(range(len(CE)), [ce[0] for ce in CE], label="ACC")
    plt.plot(range(len(CE)), [ce[1] for ce in CE], label="CE")
    plt.legend()
    plt.show()
    return


if __name__ == "__main__":
    task1()
    # test_vector_1 = []
    # predict()
    task2()
    # test_vector_2 = []
    # predict()