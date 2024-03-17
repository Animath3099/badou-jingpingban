import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt

# 基于pytorch的网络编写，实现一个网络完成一个简单nlp任务
# 随机从字表选取sentence_length = 7个字，可能重复。
# 实现一个八分类任务，如果字符'a'出现在样本中第一个位置时为第一类，
# 出现在第二个位置时为第二类...出现在第七个位置时为第七类，没出现在样本中为第八类。

class TorchModel(nn.Module):
    def __init__(self, vector_dim, sentence_length, vocab):
        super().__init__()
        self.embedding = nn.Embedding(len(vocab), vector_dim)
        # pool
        #self.pool = nn.AvgPool1d(sentence_length)
        # rnn
        self.rnn = nn.RNN(vector_dim, vector_dim, batch_first=True)
        self.classify = nn.Linear(vector_dim, sentence_length+1)
        self.loss = nn.functional.cross_entropy

    def forward(self, x, y=None):
        x = self.embedding(x)           #(batch_size, sen_len) -> (batch_size, sen_len, vector_dim)             20*7 -> 20*7*20
        #print("1:", x.shape)
        # pool
        #x = x.transpose(1,2)            #(batch_size, sen_len, vector_dim) -> (batch_size, vector_dim, sen_len) 20*7*20 -> 20*20*7
        #x = self.pool(x)                #(batch_size, vector_dim, sen_len) -> (batch_size, vector_dim, 1)       20*20*7 -> 20*20*1
        #x = x.squeeze()                 #(batch_size, vector_dim, 1) -> (batch_size, vector_dim)                20*20*1 -> 20*20
        # rnn
        rnn_out, hidden = self.rnn(x)   #(batch_size, sen_len, vector_dim) -> (batch_size, sen_len, vector_dim) 20*7*20 -> 20*7*20, 1*20*20
        #print("2:", rnn_out.shape)
        #print("3:", hidden.shape)
        #x = rnn_out[:,-1,:]             #(batch_size, sen_len, vector_dim) -> (batch_size, vector_dim)          20*7*20 -> 20*20
        x = hidden.squeeze()            #(1, batch_size, vector_dim) -> (batch_size, vector_dim)                1*20*20 -> 20*20
        #print("4:", x.shape)
        y_pred = self.classify(x)       #(batch_size, vector_dim) -> (batch_size, sen_len+1)                    20*20 -> 20*8
        #print("5:", y_pred.shape)
        if y is not None:
            return self.loss(y_pred, y)
        else:
            return y_pred

def build_vocab():
    chars = "abcdefghij"
    vocab = {"pad":0}
    for index, char in enumerate(chars):
        vocab[char] = index+1
    vocab['unk'] = len(vocab)
    return vocab

def build_sample(vocab, sentence_length):
    # 随机从字表选取sentence_length个字，可能重复
    x = [random.choice(list(vocab.keys())) for _ in range(sentence_length)]
    #print("random choice x:", x)
    # 字符'a'出现在第一个位置时为第一类，出现在第二个位置为第二类...
    # 出现在第七个位置为第七类，字符'a'没出现为第八类
    if x[0] == 'a':
        y = 0
    elif x[1] == 'a':
        y = 1
    elif x[2] == 'a':
        y = 2
    elif x[3] == 'a':
        y = 3
    elif x[4] == 'a':
        y = 4
    elif x[5] == 'a':
        y = 5
    elif x[6] == 'a':
        y = 6
    else:
        y = 7

    x = [vocab.get(word, vocab['unk']) for word in x]
    return x, y

def build_dataset(sample_length, vocab, sentence_length):
    dataset_x = []
    dataset_y = []
    for i in range(sample_length):
        x, y = build_sample(vocab, sentence_length)
        dataset_x.append(x)
        dataset_y.append(y)
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)

def build_model(vocab, char_dim, sentence_length):
    model = TorchModel(char_dim, sentence_length, vocab)
    return model

def evaluate(model, vocab, sample_length):
    model.eval()
    x, y = build_dataset(200, vocab, sample_length)  # 建立200个用于测试的样本
    #print("evaluate x:", x, x.shape)
    #print("evaluate y:", y, y.shape)
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)
        print("y_pred:", y_pred, y_pred.shape)
        for y_p, y_t in zip(y_pred, y):
            #print("y_p:", y_p, y_p.shape)
            #print("np.argmax(y_p):", np.argmax(y_p))
            #print("y_t:", y_t)
            if np.argmax(y_p) == y_t:# and y_t != 7:
                correct += 1
            else:
                wrong += 1
    print("正确预测个数：%d, 正确率：%f" %(correct, correct / (correct+wrong)))
    return correct / (correct+wrong)

def main():
    epoch_num = 20         # 训练轮数
    batch_size = 20        # 每次训练样本个数
    train_sample = 500     # 每轮训练总共训练的样本总数
    char_dim = 20          # 每个字的维度
    sentence_length = 7    # 样本文本长度
    learning_rate = 0.002  # 学习率

    vocab = build_vocab()  # 建立字表
    model = build_model(vocab, char_dim, sentence_length)             # 建立模型
    optim = torch.optim.Adam(model.parameters(), lr = learning_rate)  # 选择优化器
    log = []
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch in range(train_sample // batch_size):
            x, y = build_dataset(batch_size, vocab, sentence_length)
            optim.zero_grad()  # 梯度归零
            loss = model(x, y) # 计算loss
            loss.backward()    # 计算梯度
            optim.step()       # 更新权重
            watch_loss.append(loss.item())
            #print("x:", x)
            #print("y:", y)
        print("==============\n第%d轮平均loss:%f"%(epoch+1, np.mean(watch_loss)))
        acc = evaluate(model, vocab, sentence_length)
        log.append([acc, np.mean(watch_loss)])
    #画图
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")   #画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  #画loss曲线
    plt.legend()
    plt.show()

    torch.save(model.state_dict(), "model.pth")
    writer = open("vocab.json", "w", encoding="utf-8")
    writer.write(json.dumps(vocab, ensure_ascii=False, indent=2))
    writer.close()
    return 

def predict(model_path, vocab_path, input_strings):
    char_dim = 20
    sentence_length = 7
    vocab = json.load(open(vocab_path, "r", encoding="utf8"))  # 加载字符表
    model = build_model(vocab, char_dim, sentence_length)       # 建立模型
    model.load_state_dict(torch.load(model_path))               # 加载训练好的权重
    x = []
    for input_string in input_strings:
        x.append([vocab[char] for char in input_string])  #将输入序列化
    model.eval()
    with torch.no_grad():
        result = model.forward(torch.LongTensor(x))
    for i, input_string in enumerate(input_strings):
        print("输入：%s, 预测类别：%s, 概率值：%s" % (input_string, np.argmax(result[i]), result[i])) #打印结果

if __name__ == "__main__":
    main()
    test_strings = ["fcehedi", "afhegdc", "degaccd", "bddhjic"]
    predict("model.pth", "vocab.json", test_strings)
