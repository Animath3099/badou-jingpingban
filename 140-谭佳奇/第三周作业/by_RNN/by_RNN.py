#coding:utf8

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

"""

任务目标：返回一个字符串中a的下标（a存在且唯一）

"""

class TorchModel(nn.Module):
    def __init__(self, vector_dim, sentence_length, vocab, hidden_size):
        super(TorchModel, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(len(vocab), vector_dim)  #embedding层
        self.layer = nn.RNN(vector_dim, hidden_size, bias=False, batch_first=True)  # RNN的输入维度为词向量的长度维度，输出维度为隐藏层大小
        self.relu = nn.ReLU()
        self.linear = nn.Linear(hidden_size, sentence_length+1)
        self.activation = torch.softmax
        self.loss = nn.functional.cross_entropy

    #当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        batch_size = x.size(0)
        hidden = self.init_hidden(batch_size)
        x = self.embedding(x)                      #(batch_size, sen_len) -> (batch_size, sen_len, vector_dim)
        #x = x.transpose(1, 2)                      #(batch_size, sen_len, vector_dim) -> (batch_size, vector_dim, sen_len)
        #x = self.pool(x)                           #(batch_size, vector_dim, sen_len)->(batch_size, vector_dim, 1)
        x, hidden = self.layer(x, hidden)                           #(batch_size, sen_len, vector_dim) -> (batch_size, sen_len, hidden_size)
        x = self.relu(x)
        x = self.linear(x[:,-1,:])
        #x = x.squeeze()                            #(batch_size, vector_dim, 1) -> (batch_size, vector_dim)
        #x = self.classify(x)                       #(batch_size, vector_dim) -> (batch_size, 1) 3*5 5*1 -> 3*1
        y_pred = self.activation(x, dim=1)                #(batch_size, 1) -> (batch_size, 1)
        if y is not None:
            return self.loss(y_pred, y)   #预测值和真实值计算损失
        else:
            return y_pred                 #输出预测结果
        
    def init_hidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_size)

#字符集随便挑了一些字，实际上还可以扩充
#为每个字生成一个标号
#{"a":1, "b":2, "c":3...}
#abc -> [1,2,3]
def build_vocab():
    chars = "abcdefghij"  #字符集
    vocab = {"pad":0}
    for index, char in enumerate(chars):
        vocab[char] = index+1   #每个字对应一个序号
    vocab['unk'] = len(vocab) #11
    return vocab

#随机生成一个样本
#从所有字中选取sentence_length个字
#反之为负样本
def build_sample(vocab, sentence_length):
    #随机从字表选取sentence_length个字，可能重复但保证a出现且只出现一次
    temp_list = list(vocab.keys())
    temp_list.remove("a")
    # 建立一个长度为sentence_length-1的随机句子x，x不包含"a"
    x = [random.choice(temp_list) for _ in range(sentence_length-1)]
    # 在x中的随机位置插入"a"
    random_index = random.randrange(len(x) + 1)
    x.insert(random_index, "a")
    # temp = -1
    # 找到句子中a的下标的位置
    # for i in range(len(x)):
    #     if x[i] == "a":
    #         temp = i
    # 句子中a的位置即为random_index
    y = np.zeros(sentence_length+1)
    y[random_index] = 1
    x = [vocab.get(word, vocab['unk']) for word in x]   #将字转换成序号，为了做embedding
    return x, y

#建立数据集
#输入需要的样本数量。需要多少生成多少
def build_dataset(sample_length, vocab, sentence_length):
    dataset_x = []
    dataset_y = []
    for i in range(sample_length):
        x, y = build_sample(vocab, sentence_length)
        dataset_x.append(x)
        dataset_y.append(y)
    return torch.LongTensor(dataset_x), torch.FloatTensor(dataset_y)

#建立模型
def build_model(vocab, char_dim, sentence_length, hidden_size):
    model = TorchModel(char_dim, sentence_length, vocab, hidden_size)
    return model

#测试代码
#用来测试每轮模型的准确率
def evaluate(model, vocab, sample_length):
    model.eval()
    x, y = build_dataset(200, vocab, sample_length)   #建立200个用于测试的样本
    #print("本次预测集中共有%d个正样本，%d个负样本"%(sum(y), 200 - sum(y)))
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)      #模型预测
        for y_p, y_t in zip(y_pred, y):  #与真实标签进行对比
            if torch.argmax(y_p) == torch.argmax(y_t):
                correct += 1
            else:
                wrong += 1
    print("正确预测个数：%d, 正确率：%f"%(correct, correct/(correct+wrong)))
    return correct/(correct+wrong)


def main():
    #配置参数
    epoch_num = 50        #训练轮数
    batch_size = 20       #每次训练样本个数
    train_sample = 500    #每轮训练总共训练的样本总数
    char_dim = 20         #每个字的维度
    hidden_size = 128       # rnn隐藏层的大小
    sentence_length = 9   #样本文本长度
    learning_rate = 0.005 #学习率
    # 建立字表
    vocab = build_vocab()
    # 建立模型
    model = build_model(vocab, char_dim, sentence_length, hidden_size)
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    # 训练过程
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch in range(int(train_sample / batch_size)):
            x, y = build_dataset(batch_size, vocab, sentence_length) #构造一组训练样本
            optim.zero_grad()    #梯度归零
            loss = model(x, y)   #计算loss
            loss.backward()      #计算梯度
            optim.step()         #更新权重
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model, vocab, sentence_length)   #测试本轮模型结果
        log.append([acc, np.mean(watch_loss)])
    #画图
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  #画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  #画loss曲线
    plt.legend()
    plt.show()
    #保存模型
    torch.save(model.state_dict(), "model.pth")
    # 保存词表
    writer = open("vocab.json", "w", encoding="utf8")
    writer.write(json.dumps(vocab, ensure_ascii=False, indent=2))
    writer.close()
    return

#使用训练好的模型做预测
def predict(model_path, vocab_path, input_strings):
    char_dim = 20  # 每个字的维度
    sentence_length = 9  # 样本文本长度
    hidden_size = 128
    vocab = json.load(open(vocab_path, "r", encoding="utf8")) #加载字符表
    model = build_model(vocab, char_dim, sentence_length, hidden_size)     #建立模型
    model.load_state_dict(torch.load(model_path))             #加载训练好的权重
    x = []
    for input_string in input_strings:
        x.append([vocab[char] for char in input_string])  #将输入序列化
    model.eval()   #测试模式
    with torch.no_grad():  #不计算梯度
        result = model.forward(torch.LongTensor(x))  #模型预测
    for i, input_string in enumerate(input_strings):
        #print("输入：%s, 预测类别：%d, 概率值：%f" % (input_string, round(float(result[i])), result[i])) #打印结果
        #print(torch.argmax(result[i]).item())
        #print(torch.max(result[i]).item())
        print("输入：%s, 预测类别：%d, 概率值：%f" % (input_string, torch.argmax(result[i]).item(), torch.max(result[i]).item())) #打印结果



if __name__ == "__main__":
    main()
    test_strings = ["fbabgcaee", "adeafbdfg", "bcgfehiad", "jacbcjdef"]
    predict("model.pth", "vocab.json", test_strings)