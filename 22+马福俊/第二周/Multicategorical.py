import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


class MultiTorchmodel(nn.Module):
    def __init__(self, input_size):
        super(MultiTorchmodel, self).__init__()
        self.linear = nn.Linear(input_size, 3)  # 线性层
        self.activation = nn.Softmax(dim=1)
        self.loss = nn.functional.cross_entropy

    def forward(self, x, y=None):
        y_pre = self.linear(x)
        y_pred = self.activation(y_pre)
        if y is not None:
            return self.loss(y_pre, y)
        else:
            return y_pred


def build_sample():
    x = np.random.random(5)
    # 求均值范围
    mean_num = np.mean(x)
    if mean_num < 0:
        return x, 0
    elif mean_num > 0 and mean_num < 0.5:
        return x, 1
    else:
        return x, 2


def build_dataset(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append(y)
    return torch.FloatTensor(X), torch.LongTensor(Y)


def evaluate(model):
    model.eval()
    test_sample_num = 100
    x, y = build_dataset(test_sample_num)
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)  # 模型预测
        _, predicted = torch.max(y_pred, 1)
        for y_p, y_t in zip(predicted, y):
            # print("预测值：%s, 真实值：%s" % (y_p.item(), y_t.item()))

            if y_p.item() == y_t.item():
                correct += 1
            else:
                wrong += 1
    print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)

def main():
    # 配置参数
    epoch_num = 20  # 训练轮数
    batch_size = 20  # 每次训练样本个数
    train_sample = 5000  # 每轮训练总共训练的样本总数
    input_size = 5  # 输入向量维度
    learning_rate = 0.001  # 学习率
    # 建立模型
    model = MultiTorchmodel(input_size)
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


def predict(model_path, input_vec):
    input_size = 5
    torchmodel = MultiTorchmodel(input_size)
    torchmodel.load_state_dict(torch.load(model_path))

    torchmodel.eval()
    with torch.no_grad():
        result = torchmodel.forward(torch.FloatTensor(input_vec))
        _, predicted = torch.max(result, 1)
        for vec, res in zip(input_vec, predicted):
            print("输入：%s, 预测类别：%s" % (vec, predicted))


if __name__ == "__main__":
    main()
    test_vec = [[0.47889086, 0.15229675, 0.31082123, 0.03504317, 0.18920843],
                [0.94963533, 0.5524256, 0.95758807, 0.95520434, 0.84890681],
                [0.78797868, 0.67482528, 0.13625847, 0.34675372, 0.09871392],
                [0.89349776, 0.59416669, 0.92579291, 0.41567412, 0.7358894]]
    predict("model.pt", test_vec)
