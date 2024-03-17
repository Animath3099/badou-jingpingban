"""
基于pytorch框架编写模型训练
实现一个自行构造的找规律(机器学习)任务
五维判断：x是一个5维向量，向量中哪个标量最大就输出哪一维下标

第二周作业：
规律：x是一个5维向量，输出一个5维向量，若x最大值是下标为x[3],则y[3]=1,其他为0，以此类推
[1,2,5,3,4] => [0,0,1,0,0]
[1,2,3,4,5] => [0,0,0,0,1]
[5,4,3,2,1] => [1,0,0,0,0]
test commit to the main branch
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

class MultiClassficationModel(nn.Module):
    def __init__(self, input_size):
        super(MultiClassficationModel, self).__init__()
        self.linear = nn.Linear(input_size, input_size)  # 输出维度与输入维度相同
        self.loss = nn.CrossEntropyLoss()  # 使用交叉熵损失函数

    def forward(self, x, y=None):
        y_pred = self.linear(x)
        if y is not None:
            return self.loss(y_pred, y)  # 计算损失
        else:
            return y_pred  # 返回预测值

def build_sample():
    x = np.random.rand(5) * 10  # 生成随机向量
    y = np.argmax(x)  # 获取最大值的索引作为标签
    return x, y

def build_dataset(total_sample_num):
    X, Y = [], []
    for _ in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append(y)
    return torch.FloatTensor(np.array(X)), torch.LongTensor(np.array(Y))



def evaluate(model, test_sample_num=100):
    model.eval()
    x, y = build_dataset(test_sample_num)
    correct = 0
    with torch.no_grad():
        y_pred = model(x)
        for y_p, y_t in zip(y_pred, y):
            if torch.argmax(y_p) == torch.argmax(y_t):
                correct += 1

    accuracy = correct / test_sample_num
    print(f"正确预测个数：{correct}, 正确率：{accuracy}")
    return accuracy

def main():
    epoch_num = 20
    batch_size = 20
    train_sample = 5000
    input_size = 5
    learning_rate = 0.001

    model = MultiClassficationModel(input_size)
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)

    log = []
    train_x, train_y = build_dataset(train_sample)

    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch_index in range(train_sample // batch_size):
            x = train_x[batch_index * batch_size: (batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size: (batch_index + 1) * batch_size]
            loss = model(x, y)
            loss.backward()
            optim.step()
            optim.zero_grad()
            watch_loss.append(loss.item())

        print(f"第{epoch + 1}轮平均loss: {np.mean(watch_loss)}")
        acc = evaluate(model)
        log.append([acc, np.mean(watch_loss)])

    torch.save(model.state_dict(), "model.pt")
    plt.plot(range(len(log)), [l[0] for l in log], label="Accuracy")
    plt.plot(range(len(log)), [l[1] for l in log], label="Loss")
    plt.legend()
    plt.show()

# 使用训练好的模型做预测
def predict(model_path, input_vec):
    input_size = 5
    model = MultiClassficationModel(input_size)
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重
    print(model.state_dict())

    model.eval()  # 测试模式
    input_tensor = torch.FloatTensor(np.array([input_vec]))  # 修改这里
    with torch.no_grad():  # 不计算梯度
        result = model(input_tensor)  # 模型预测
    for vec, res in zip([input_vec], result):
        print("输入：%s, 预测类别：%s, 概率值：%s" % (vec, torch.argmax(res), res))  # 打印结果



if __name__ == "__main__":
    # main()
    test_vector = np.random.rand(5) * 10
    print("测试向量:", test_vector)
    predict("model.pt", [test_vector])
