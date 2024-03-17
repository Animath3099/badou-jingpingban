#coding:utf8
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt
class TorchModel(nn.Module):
    def __init__(self, vector_dim, sentence_length, vocab):
        super(TorchModel, self).__init__()
        self.embedding = nn.Embedding(len(vocab), vector_dim)

        self.lstm = nn.LSTM(vector_dim, vector_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.5)  # 添加 Dropout 层

        self.classify = nn.Linear(vector_dim * 2, out_features=1)
        self.activation = torch.sigmoid

        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, x, y=None):
        x = self.embedding(x)
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.dropout(x)
        x = self.classify(x)
        y_pred = self.activation(x)
        if y is not None:
            y = y.unsqueeze(1)  # 添加额外的维度
            return self.loss(y_pred, y)
        else:
            return y_pred
def build_vocab():
    chars = "abcdefghijklmnopqrstuvwxyz"
    vocab = {"pad":0}
    for index, char in enumerate(chars):
        vocab[char] = index+1
    vocab['unk'] = len(vocab)
    return vocab

def build_sample(vocab, sentence_length):
    input_string = []
    if random.random() < 0.5:
        input_string.append("a")
    l = random.randint(3, sentence_length)
    input_string += [random.choice(list(vocab.keys())) for _ in range(l)]

    # 截断或填充输入字符串
    if len(input_string) > sentence_length:
        input_string = input_string[:sentence_length]  # 截断过长的字符串
    elif len(input_string) < sentence_length:
        input_string += ['pad'] * (sentence_length - len(input_string))  # 使用 'pad' 填充过短的字符串
    x = [vocab.get(char, vocab['pad']) for char in input_string]
    if input_string[0] == "a":
        y = 1
    else:
        y = 0
    return x, y
def build_dataset(sample_length, vocab, sentence_length):
    dataset_x, dataset_y = [], []
    for i in range(sample_length):
        x, y = build_sample(vocab, sentence_length)
        dataset_x.append(x)
        dataset_y.append(y)
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)
    pass
def build_model(vocab, char_dim, sentence_length):
    model = TorchModel(char_dim, sentence_length, vocab)
    return model
    pass

def evaluate(model, vocab, sample_length):
    model.eval()
    x, y =build_dataset(100, vocab, sample_length)
    print(x)
    print(y)
    print("本次预测集中共有%d个正样本，%d个负样本"%(sum(y), 100 - sum(y)))
    correct, wrong =0, 0
    with torch.no_grad():
        y_pred = model(x)
        for y_p, y_t in zip(y_pred, y):
            if float(y_p) < 0.5 and int(y_t) == 0:
                correct += 1
            elif float(y_p) >= 0.5 and int(y_t) == 1:
                correct += 1
            else:
                wrong += 1
    print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)
def main():
    epoch_num = 5
    batch_size = 20
    train_sample = 500
    char_dim = 20
    sentence_length = 6
    learning_rate = 0.005
    #数据预处理
    vocab = build_vocab()
    #初始化建立模型
    model = build_model(vocab, char_dim, sentence_length)
    #选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch in range(int(train_sample / batch_size)):
            x, y = build_dataset(batch_size, vocab, sentence_length)
            y = y.float()
            optim.zero_grad()
            loss = model(x, y)
            loss.backward()
            optim.step()
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model, vocab, sentence_length)
        log.append([acc, np.mean(watch_loss)])
    # 画图
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    plt.legend()
    plt.savefig('123')
    # 保存模型
    torch.save(model.state_dict(), "model.pth")
    # 保存词表
    writer = open("vocab.json", "w", encoding="utf8")
    writer.write(json.dumps(vocab, ensure_ascii=False, indent=2))
    writer.close()

def predict(model_path, vocab_path, input_strings):
    char_dim = 20
    sentence_length = 6
    vocab = json.load(open(vocab_path, "r", encoding="utf8"))
    model = build_model(vocab, char_dim, sentence_length)
    model.load_state_dict(torch.load(model_path))
    x = []
    for input_string in input_strings:
        # 截断或填充输入字符串
        if len(input_string) > sentence_length:
            input_string = input_string[:sentence_length]  # 截断过长的字符串
        elif len(input_string) < sentence_length:
            input_string += ('0'* (sentence_length - len(input_string)))  # 使用 '0' 填充过短的字符串

        x.append([vocab.get(char, vocab['unk']) for char in input_string])

    model.eval()
    with torch.no_grad():
        result = model.forward(torch.LongTensor(x))
    for i, input_string in enumerate(input_strings):
        print("输入：%s, 预测类别：%d, 概率值：%f" % (input_string, round(float(result[i])), result[i])) #打印结果

if __name__ == "__main__":
    main()
    test_strings = [
        "abandon",
        "bat",
        "fever",
        "long"
    ]
    predict("model.pth", "vocab.json", test_strings)
