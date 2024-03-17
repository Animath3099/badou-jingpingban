# coding:utf8

import torch
import torch.nn as nn
import numpy as np
import random
import json
from transformers import BertModel

"""

基于pytorch的网络编写
实现一个网络完成一个简单nlp任务
判断文本中是否有某些特定字符出现

week2的作业，修改引入bert
"""


class TorchModel(nn.Module):
    def __init__(self, input_dim):
        super(TorchModel, self).__init__()
        self.bert = BertModel.from_pretrained(r"bert-base-chinese", return_dict=False)
        self.classify = nn.Linear(self.state_dict()['bert.embeddings.word_embeddings.weight'].shape[1], 3)
        self.activation = torch.sigmoid  # sigmoid做激活函数
        self.dropout = nn.Dropout(0.5)
        self.loss = nn.functional.cross_entropy

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        sequence_output, pooler_output = self.bert(x)

        x = self.classify(pooler_output)
        y_pred = self.activation(x)
        if y is not None:
            return self.loss(y_pred, y.squeeze())
        else:
            return y_pred


# 字符集随便挑了一些汉字，实际上还可以扩充
# 为每个字生成一个标号
# {"a":1, "b":2, "c":3...}
# abc -> [1,2,3]
def build_vocab():
    chars = "abcdefghijklmnopqrstuvwxyz"  # 字符集
    vocab = {}
    for index, char in enumerate(chars):
        vocab[char] = index + 1  # 每个字对应一个序号
    vocab['unk'] = len(vocab) + 1
    return vocab


# 随机生成一个样本
# 从所有字中选取sentence_length个字
# 反之为负样本
def build_sample(vocab, sentence_length):
    # 随机从字表选取sentence_length个字，可能重复
    x = [random.choice(list(vocab.keys())) for _ in range(sentence_length)]
    # A类样本
    if set("abc") & set(x) and not set("xyz") & set(x):
        y = 0
    # B类样本
    elif not set("abc") & set(x) and set("xyz") & set(x):
        y = 1
    # C类样本
    else:
        y = 2
    x = [vocab.get(word, vocab['unk']) for word in x]  # 将字转换成序号，为了做embedding
    return x, y


# 建立数据集
# 输入需要的样本数量。需要多少生成多少
def build_dataset(sample_length, vocab, sentence_length):
    dataset_x = []
    dataset_y = []
    for i in range(sample_length):
        x, y = build_sample(vocab, sentence_length)
        dataset_x.append(x)
        dataset_y.append([y])
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)


# 建立模型
def build_model(vocab, char_dim, sentence_length):
    model = TorchModel(char_dim)
    return model


# 测试代码
# 用来测试每轮模型的准确率
def evaluate(model, vocab, sample_length):
    model.eval()
    total = 200  # 测试样本数量
    x, y = build_dataset(total, vocab, sample_length)  # 建立200个用于测试的样本
    y = y.squeeze()
    print("A类样本数量：%d, B类样本数量：%d, C类样本数量：%d" % (
        y.tolist().count(0), y.tolist().count(1), y.tolist().count(2)))
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)  # 模型预测
        for y_p, y_t in zip(y_pred, y):  # 与真实标签进行对比
            if int(torch.argmax(y_p)) == int(y_t):
                correct += 1  # 正样本判断正确
            else:
                wrong += 1
    print("正确预测个数：%d / %d, 正确率：%f" % (correct, total, correct / (correct + wrong)))
    return correct / (correct + wrong)


def main():
    epoch_num = 1  # 训练轮数
    batch_size = 10  # 每次训练样本个数
    train_sample = 10  # 每轮训练总共训练的样本总数
    char_dim = 768  # 每个字的维度
    sentence_length = 1  # 样本文本长度
    vocab = build_vocab()  # 建立字表
    model = build_model(vocab, char_dim, sentence_length)  # 建立模型
    optim = torch.optim.Adam(model.parameters(), lr=1e-5)  # 建立优化器
    log = []
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch in range(int(train_sample / batch_size)):
            x, y = build_dataset(batch_size, vocab, sentence_length)  # 构建一组训练样本
            optim.zero_grad()  # 梯度归零
            loss = model(x, y)  # 计算loss
            loss.backward()  # 计算梯度
            optim.step()  # 更新权重
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model, vocab, sentence_length)  # 测试本轮模型结果
        log.append([acc, np.mean(watch_loss)])
    # plt.plot(range(len(log)), [l[0] for l in log])  #画acc曲线
    # plt.plot(range(len(log)), [l[1] for l in log])  #画loss曲线
    # plt.show()
    # 保存模型
    torch.save(model.state_dict(), "model.pth")
    count = 0
    for key, value in model.state_dict().items():
        # 计算当前张量的参数个数
        # print(f'layer name: {key},            layer shape: {value.numel()}')
        print(f'layer name: {key},            layer shape: {value.shape}')
        # print(f'layer value: {value}')
        count += value.numel()
    else:
        print(f'参数总量为：{count}')


if __name__ == "__main__":
    main()
    # test_strings = ["juvaee", "yrwfrg", "rtweqg", "nlhdww"]
    # predict("model.pth", "vocab.json", test_strings)

    d_model = 768
    vocab_size = 21128
    max_position_embeddings = 512
    token_type = 2
    # 嵌入层参数量
    embedding_layer_params = vocab_size * d_model + max_position_embeddings * d_model + token_type * d_model + 2 * d_model

    # 自注意力层参数量
    self_attention_params = (d_model * d_model + d_model) * 3 + d_model * d_model + d_model
    # 归一化层参数量
    norm1 = 768 * 2
    # feed foward层参数量
    feed_foward_params = d_model * 4 * d_model + 4 * d_model + 4 * d_model * d_model + d_model
    # 归一化层参数量
    norm2 = 768 * 2

    # encode层数
    encode_num = 3
    # encode层参数总量
    encode_params = encode_num * (self_attention_params + norm1 + feed_foward_params + norm2)

    # pool层参数总量
    pool_params = d_model * d_model + d_model

    # bert总参数量
    bert_params = embedding_layer_params + encode_params + pool_params

    classify_params = d_model * 3 + 3
    print(f'手动计算的参数总量为：{bert_params + classify_params}')
