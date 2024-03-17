#coding:utf8

import torch
import torch.nn as nn
import numpy as np
import random
import json

class Nlpmodel(nn.Module):
    def __init__(self, vector_dim, sentence_length, vocab):
        super(Nlpmodel, self).__init__()
        self.embedding = nn.Embedding(len(vocab), vector_dim)
        # self.pool = nn.AvgPool1d(sentence_length)
        self.rnn_layer = nn.RNN(input_size=vector_dim,
                            hidden_size=vector_dim,
                            batch_first=True,
                            )
        self.classify = nn.Linear(vector_dim, sentence_length+1)
        # self.activation = torch.sigmoid
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, y=None):
        x = self.embedding(x)                          # (batch_size, sen_len) -> (batch_size, sen_len, vector_dim)
        x, _ = self.rnn_layer(x)                  # (batch_size, sen_len, vector_dim)  -> (batch_size, sen_len, hidden_size)
        x = x[:, -1, :]                           # 降维操作，将3维降为2维  这里是把中间那一维取消
        y_pred = self.classify(x)
        if y is not None:
            y = y.squeeze().long()
            return self.loss(y_pred, y)
        else:
            return y_pred


# 构建一个词表
def build_dict():
    chars = "abcdefghijk"
    vocab = {}
    for index, char in enumerate(chars):
        vocab[char] = index
    vocab["unk"] = len(chars)
    return vocab

# 创建一个样本
def build_sample(vocab, sentence_length):
    x = random.sample(list(vocab.keys()), sentence_length)
    # 正样本
    if "a" in x:
        y = x.index("a")
    # 负样本
    else:
        y = sentence_length
    x = [vocab.get(word, vocab["unk"]) for word in x]
    return x, y

# 通过样本创建数据集
def build_dataset(num_sample, vocab, sentence_length):
    data_x = []
    data_y = []
    for i in range(num_sample):
        x, y = build_sample(vocab, sentence_length)
        data_x.append(x)
        data_y.append([y])
    return torch.LongTensor(data_x), torch.FloatTensor(data_y)

# 建立模型
def build_model(vocab, char_dim, sentence_length):
    model = Nlpmodel(char_dim, sentence_length, vocab)
    return model

# 模型测试，测试每轮的准确率
def evluate(model, vocab, sample_length):
    model.eval()
    x, y = build_dataset(200, vocab, sample_length)
    n_sample = list(y).count(sample_length)    # 统计负样本的个数
    p_sample = 200 - n_sample           # 统计正样本的个数
    print("测试样本中有%d个正样本， %d个负样本" % (p_sample, n_sample))
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)
        for y_p, y_t in zip(y_pred, y):
            if int(torch.argmax(y_p)) == int(y_t):
                correct += 1
            else:
                wrong += 1
    print("模型预测的正确数为：%d, 准确率为：%f" % (correct, correct/(correct+wrong)))
    return correct/(correct+wrong)


# 主函数
def main():
    # 定义参数
    epoch_num = 20
    train_sample = 1000
    batch_size = 20
    char_dim = 20
    sentence_length = 6
    learning_rate = 0.005
    # hidden_size = 64

    # 定义相关函数
    vocab = build_dict()
    model = build_model(vocab, char_dim, sentence_length)
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # 训练过程
    for epoch in range(epoch_num):
        model.train()
        num_loss = []
        for batch in range(int(train_sample / batch_size)):
            x, y = build_dataset(batch_size, vocab, sentence_length)
            optim.zero_grad()
            loss = model(x, y)
            loss.backward()
            optim.step()
            num_loss.append(loss.item())
        print("-------第%d轮的平均loss: %s" % (epoch + 1, np.mean(num_loss)))
        acc = evluate(model, vocab, sentence_length)

    # 保存模型
    torch.save(model.state_dict(), "model5.pth")
    # 保存词表
    writer = open("vocab5.json", "w", encoding="utf8")
    writer.write(json.dumps(vocab, ensure_ascii=False, indent=2))
    writer.close()
    return

if __name__ == "__main__":
    main()