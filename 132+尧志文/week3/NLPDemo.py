import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt
import random
import string
emb = nn.Embedding(26, 5)

def build_set(n):
    x = []
    y_true = []
    word = "bcdefghijklmnopqrstuvwxyz"
    for i in range(n):
        beta1 = random.uniform(0, 1)
        pre = []
        if beta1 < 1/6:
            res = random.choices(list(word), k=5)
            y = 5
        if 1/6 < beta1 < 2/6:
            res = random.choices(list(word), k=4)
            res.insert(0, "a")
            y = 0
        if 2/6 < beta1 < 3/6:
            res = random.choices(list(word), k=4)
            res.insert(1, "a")
            y = 1
        if 3/6 < beta1 < 4/6:
            res = random.choices(list(word), k=4)
            res.insert(2, "a")
            y = 2
        if 4/6 < beta1 < 5/6:
            res = random.choices(list(word), k=4)
            res.insert(3, "a")
            y = 3
        if 5 / 6 < beta1 < 1:
            res = random.choices(list(word), k=4)
            res.insert(4, "a")
            y = 4
        for a in res:
            pre.append(ord(a) - 97)
        x.append(pre)
        y_true.append([y])
    return torch.LongTensor(x), torch.FloatTensor(y_true)

x, y = build_set(5)
print(x)
print(emb(x))
print(y)
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.emb = nn.Embedding(26, 5)
        self.pool = nn.AvgPool1d(5)
        self.rnn = nn.RNN(5, 10, bias=False, batch_first=True)
        self.linear = nn.Linear(10, 6)
        self.relu = nn.ReLU()
        self.loss = nn.CrossEntropyLoss()
        self.fc = nn.Sequential(self.relu, self.linear)
    def forward(self, x, y=None):
        out1 = self.emb(x)
        out2, out3= self.rnn(out1)
        y_pred = torch.squeeze(self.fc(out3))
        if y is not None:
            return self.loss(y_pred, y)
        else:
            return y_pred

def evaluate(model):
    model.eval()
    test_sample_num = 100
    x, y = build_set(test_sample_num)
    print("本次预测集中共有%d个0样本，%d个1样本, %d个2样本，%d个3样本，%d个4样本, %d个5样本" % (
    torch.flatten(y).tolist().count(0), torch.flatten(y).tolist().count(1), torch.flatten(y).tolist().count(2),
    torch.flatten(y).tolist().count(3), torch.flatten(y).tolist().count(4), torch.flatten(y).tolist().count(5)))
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)
        for y_p, y_t in zip(y_pred, y):  # 与真实标签进行对比
            if int(np.argmax(y_p)) == int(y_t):
                correct += 1  # 判断正确
            else:
                wrong += 1
    print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)

def main():
    epoch_num = 40
    batch_size = 20
    train_sample = 5000
    learning_rate = 0.001
    model = Model()
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch in range(int(train_sample / batch_size)):
            x, y_true = build_set(batch_size)
            y = torch.flatten(y_true).long()
            optim.zero_grad()
            loss = model(x, y)
            loss.backward()
            optim.step()
            optim.zero_grad()
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
def trans(list):
    ans = []
    for i in list:
        pre = []
        for a in i:
            pre.append(ord(a) - 97)
        ans.append(pre)
    return torch.LongTensor(ans)

def predict(model_path, input_vec, input_tensor):
    model = Model()
    softmax = nn.Softmax(dim=-1)
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重
    print(model.state_dict())

    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        result = model.forward(torch.LongTensor(input_vec))  # 模型预测
    for vec, res in zip(input_tensor, result):
        print("输入：%s, 预测类别：%d, 概率值：%f" % (vec, int(np.argmax(res.numpy())), float(np.max(softmax(res).numpy()))))  # 打印结



if __name__ == "__main__":
    main()
    input_tensor = ["abcde", "favgh", "ohalj", "pubai", "asdfa", "lkjhg"]
    test_string = trans(input_tensor)
    predict("model.pt", test_string, input_tensor)













