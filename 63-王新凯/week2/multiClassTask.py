import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


class MultiClassfModel(nn.Module):
    def __init__(self, input_dim):
        super(MultiClassfModel, self).__init__()
        self.linear0 = nn.Linear(input_dim, input_dim)
        self.activation0 = torch.softmax
        self.loss = nn.functional.cross_entropy

    def forward(self, x, y=None):
        x = self.linear0(x)
        y_pred = self.activation0(x, dim=1)
        if y is not None:
            return self.loss(y_pred, y)
        else:
            return y_pred

# 生成dim维向量x，并返回其中元素数值最大的下标。可以看做dim-class分类任务
def genSample(input_dim):
    x = np.random.random(input_dim)
    y = x.argmax()
    return x, y


def genSamples(sample_size, input_dim):
    X = []
    Y = []
    for i in range(sample_size):
        x, y = genSample(input_dim)
        X.append(x)
        Y.append(y)
    return torch.FloatTensor(X), torch.LongTensor(Y)



dim = 20

def evaluate(model):
    model.eval()
    test_sample_num = 100
    x, y = genSamples(test_sample_num, dim)
    #print("本次预测集中共有%d个正样本，%d个负样本" % (sum(y), test_sample_num - sum(y)))
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)  # 模型预测
        for y_p, y_t in zip(y_pred, y):  # 与真实标签进行对比
            y_p_max_i = y_p.argmax()
            if y_p_max_i == y_t:
                correct += 1
            else:
                wrong += 1
    print("正确预测个数：%d, 正确率：%f" % (correct, correct / test_sample_num))
    return correct / test_sample_num


model_path = "model.pt"

def train():
    sample_size_train = 20000
    epoch = 100
    batch_size = 100
    learning_rate = .01

    x_train, y_train = genSamples(sample_size_train, dim)

    loss = 0
    # print(f"X:{X},\n Y:{Y}")
    model = MultiClassfModel(dim)
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    for epoch_i in range(epoch):
        loss_hist = []
        model.train()
        for batch_i in range(sample_size_train // batch_size):
            x_batch = x_train[batch_i * batch_size: (batch_i + 1) * batch_size]
            y_batch = y_train[batch_i * batch_size: (batch_i + 1) * batch_size]
            loss = model(x_batch, y_batch)
            loss.backward()
            optim.step()
            optim.zero_grad()
            loss_hist.append(loss.item())
        mean_loss = np.mean(loss_hist)
        print("=========\n第%d轮平均loss:%f" % (epoch_i + 1, mean_loss))
        acc = evaluate(model)  # 测试本轮模型结果
        log.append([acc, mean_loss])
    # 保存模型
    torch.save(model.state_dict(), model_path)
    # 画图
    print(log)
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.plot(range(len(log)), [l[0] for l in log], color='green', label="acc")
    ax.legend(loc="upper left")
    ax2 = ax.twinx()
    ax2.plot(range(len(log)), [l[1] for l in log], color='red', label="loss")
    ax2.legend(loc="upper right")
    plt.title(f"{dim}-class classification task, batch size({batch_size}), lr({learning_rate}), epoch({epoch})")
    plt.show()
    return


# 使用训练好的模型做预测
def predict(model_path, input_vec):
    model = MultiClassfModel(dim)
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重
    print(model.state_dict())

    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        result = model(input_vec)  # 模型预测
    for vec, res in zip(input_vec, result):
        probability = res.max()
        class_i = res.argmax()
        print("输入：%s, 预测类别：%d, 概率值：%f" % (vec, class_i, probability))  # 打印结果


if __name__ == "__main__":
    train()

    x,_ = genSamples(10, dim)

    predict(model_path, x)


