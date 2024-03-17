import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import json
import matplotlib.pyplot as plt

def build_vocab():
    chars = "abcdefghijklmnopqrstuvwxyz"  # 字符集包含所有26个字母
    vocab = {"pad": 0}
    for index, char in enumerate(chars):
        vocab[char] = index + 1   # 每个字对应一个序号
    vocab['unk'] = len(vocab)  # 27
    return vocab

def build_sample(vocab, sentence_length):
    x = [random.choice(list(vocab.keys())) for _ in range(sentence_length)]  # 随机选择字母
    y = 0
    for i in range(len(x)):
        if 'a' in x[i]:
            y = i  # 记录字母 "a" 的位置
            break
    x = [vocab.get(word, vocab['unk']) for word in x]   # 将字转换成序号，为了做embedding
    return torch.LongTensor(x), torch.LongTensor([y])  # 转换为 Tensor 类型并返回

def build_dataset(sample_length, vocab, sentence_length):
    dataset_x = []
    dataset_y = []
    for i in range(sample_length):
        x, y = build_sample(vocab, sentence_length)
        dataset_x.append(x)
        dataset_y.append(y)
    return torch.stack(dataset_x), torch.cat(dataset_y).long()  # 转换为 Tensor 类型并返回

class TorchRNN(nn.Module):
    def __init__(self, vocab, input_size, hidden_size, output_size):
        super(TorchRNN, self).__init__()
        self.embedding = nn.Embedding(len(vocab), input_size)
        self.layer = nn.RNN(input_size, hidden_size, bias=True)
        self.classify = nn.Linear(hidden_size, output_size)

    def forward(self, x, y = None):
        embedded = self.embedding(x)
        output, _ = self.layer(embedded)
        y_pred = self.classify(output[:,-1,:])
        if y is not None:
            loss = nn.functional.cross_entropy(y_pred, y, ignore_index=len(vocab))
            return y_pred, loss
        else:
            return y_pred, None
        

#建立模型
def build_model(vocab, char_dim, sentence_length, output_size):
    model = TorchRNN(vocab, char_dim, sentence_length, output_size)
    return model

def evaluate(model, vocab, sample_length):
    model.eval()
    x, y = build_dataset(200, vocab, sample_length)   # 建立200个用于测试的样本
    correct = 0
    total = 0
    with torch.no_grad():
        y_pred = model(x)      # 模型预测
        _, predicted = torch.max(y_pred, 1)  # 获取预测结果中概率最大的类别
        total += y.size(0)
        correct += (predicted == y).sum().item()  # 统计预测正确的样本数
    accuracy = correct / total
    print("正确预测个数：%d, 正确率：%f" % (correct, accuracy))
    return accuracy

def main():
    #配置参数
    epoch_num = 20        #训练轮数
    batch_size = 20       #每次训练样本个数
    train_sample = 500    #每轮训练总共训练的样本总数
    char_dim = 20         #每个字的维度
    sentence_length = 6   #样本文本长度
    learning_rate = 0.005 #学习率
    output_size = 5       #多分类标签个数
    # 建立字表
    vocab = build_vocab()
    # 建立模型
    model = build_model(vocab, char_dim, sentence_length, output_size)
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    # 训练过程
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch in range(int(train_sample / batch_size)):
            x, y = build_dataset(batch_size, vocab, sentence_length)  # 构造一组训练样本
            optim.zero_grad()  # 梯度归零
            y_pred, loss = model(x, y)  # 计算loss
            if loss is not None:
                loss.backward()  # 计算梯度
                optim.step()  # 更新权重
            watch_loss.append(loss.item())

        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model, vocab, sentence_length)   # 测试本轮模型结果
        log.append([acc, np.mean(watch_loss)])

    # 画图
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    plt.legend()
    plt.show()

    #保存模型
    torch.save(model.state_dict(), "model_RNN.pth")
    # 保存词表
    writer = open("vocab.json", "w", encoding="utf8")
    writer.write(json.dumps(vocab, ensure_ascii=False, indent=2))
    writer.close()
    return