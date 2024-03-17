# coding:utf8

import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt

"""

基于pytorch的网络编写
实现一个网络完成一个简单nlp任务
判断文本中是否有某些特定字符出现

"""


class TorchModel(nn.Module):
    def __init__(self, vector_dim, sentence_length, vocab):
        super(TorchModel, self).__init__()
        self.embedding = nn.Embedding(len(vocab) + 1, vector_dim)
        self.layer = nn.RNN(vector_dim, 10, batch_first=True)  # 线性层的一种替换
        self.classify = nn.Linear(10, sentence_length)
        self.loss = nn.CrossEntropyLoss()

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x0, y=None):
        x1 = self.embedding(x0)
        x3, x2 = self.layer(x1)
        x3 = x3[:, -1, :]
        y_pred = self.classify(x3)

        if y is not None:
            return self.loss(y_pred, y)  # 预测值和真实值计算损失
        else:
            return y_pred  # 输出预测结果


def build_vocab():
    chars = "abcdefghijklmnopqrstuvwxyz"  # 字符集
    vocab = dict()
    for index, char in enumerate(chars):
        vocab[char] = index + 1  # 每个字对应一个序号
    return vocab


# 随机生成一个样本
def build_sample(vocab, sentence_length):
    random_chars = ''.join(random.choices(list(vocab.keys()), k=sentence_length))

    # 确保随机字符串中包含字母 "a"
    if 'a' not in random_chars:
        # 在随机字符串中随机选择一个位置，将其替换为 "a"
        random_index = random.randint(0, 3)
        random_chars = random_chars[:random_index] + 'a' + random_chars[random_index + 1:]
    y = random_chars.find('a')
    x = [vocab.get(word) for word in random_chars]  # 将字转换成序号，为了做embedding
    return x, y


# 建立数据集
# 输入需要的样本数量。需要多少生成多少
def build_dataset(sample_length, vocab, sentence_length):
    dataset_x = []
    dataset_y = []
    for i in range(sample_length):
        x, y = build_sample(vocab, sentence_length)
        dataset_x.append(x)
        dataset_y.append(y)
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)


# 建立模型
def build_model(vocab, char_dim, sentence_length):
    model = TorchModel(char_dim, sentence_length, vocab)
    return model


# 测试代码
# 用来测试每轮模型的准确率
def evaluate(model, vocab, sample_length):
    model.eval()
    x, y = build_dataset(200, vocab, sample_length)  # 建立200个用于测试的样本
    print("本次预测集中共有%d个样本" % (200))
    y_true = to_one_hot(y, sample_length)
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)  # 模型预测
        for y_p, y_t in zip(y_pred, y_true):  # 与真实标签进行对比
            if int(torch.argmax(y_p)) == int(torch.argmax(y_t)):
                correct += 1  # 样本判断正确
            else:
                wrong += 1
    print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)


# 将输入转化为onehot矩阵
def to_one_hot(y, sentence_length):
    y_one_hot = torch.zeros(len(y), sentence_length)  # 创建一个全零张量，形状为(len(y), num_classes)
    # 使用scatter_函数将对应位置置为1
    y_one_hot.scatter_(1, y.unsqueeze(1), 1)
    return y_one_hot


def main():
    # 配置参数
    epoch_num = 20  # 训练轮数
    batch_size = 15  # 每次训练样本个数
    train_sample = 500  # 每轮训练总共训练的样本总数
    char_dim = 16  # 每个字的维度
    sentence_length = 6  # 样本文本长度
    learning_rate = 0.005  # 学习率
    # 建立字表
    vocab = build_vocab()
    # 建立模型
    model = build_model(vocab, char_dim, sentence_length)
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    # 训练过程
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch in range(int(train_sample / batch_size)):
            x, y = build_dataset(batch_size, vocab, sentence_length)  # 构造一组训练样本
            y = to_one_hot(y, sentence_length)
            optim.zero_grad()  # 梯度归零
            loss = model(x, y)  # 计算loss
            loss.backward()  # 计算梯度
            optim.step()  # 更新权重
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model, vocab, sentence_length)  # 测试本轮模型结果
        log.append([acc, np.mean(watch_loss)])
    # 画图
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    plt.legend()
    plt.show()
    # 保存模型
    torch.save(model.state_dict(), "model.pth")
    # 保存词表
    # writer = open("vocab.json", "w", encoding="utf8")
    # writer.write(json.dumps(vocab, ensure_ascii=False, indent=2))
    # writer.close()
    return


# 使用训练好的模型做预测
# def predict(model_path, vocab_path, input_strings):
#     char_dim = 20  # 每个字的维度
#     sentence_length = 6  # 样本文本长度
#     vocab = json.load(open(vocab_path, "r", encoding="utf8"))  # 加载字符表
#     model = build_model(vocab, char_dim, sentence_length)  # 建立模型
#     model.load_state_dict(torch.load(model_path))  # 加载训练好的权重
#     x = []
#     for input_string in input_strings:
#         x.append([vocab[char] for char in input_string])  # 将输入序列化
#     model.eval()  # 测试模式
#     with torch.no_grad():  # 不计算梯度
#         result = model.forward(torch.LongTensor(x))  # 模型预测
#     for i, input_string in enumerate(input_strings):
#         print("输入：%s, 预测类别：%d, 概率值：%f" % (input_string, round(float(result[i])), result[i]))  # 打印结果
#

if __name__ == "__main__":
    main()
    # test_strings = ["fnvfea", "awsdfg", "rqwaeg", "nakwww"]
    # predict("model.pth", "vocab.json", test_strings)
