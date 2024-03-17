import numpy as np
import torch
import torch.nn as nn
import os
import matplotlib.pyplot as plt
import random

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
"""
作业：设置一个6分类，判断a出现的位置，则表示第几类，不允许重复出现a，如a不出现则显示第6类
"""



class RnnTM(nn.Module):
    def __init__(self, sentence_length, char_dim, vocab):
        super(RnnTM, self).__init__()
        self.embedding = nn.Embedding(len(vocab), char_dim, padding_idx=0)
        self.layer = nn.RNN(char_dim, sentence_length, bias=False, batch_first=True)
        self.loss = nn.functional.cross_entropy
    def forward(self, x, y=None):
        x = self.embedding(x)
        y_1, y_2 = self.layer(x)
        y_2 = y_2.squeeze()
        if y is None:
            return y_2
        else:
            return self.loss(y_2, y)

def build_vocab():
    vocab = {'pad': 0}
    s = 'qwertyuiopasdfghjklzxcvbnm'
    num = 1
    for i in s:
        vocab[i]=num
        num+=1
    vocab['unk']=num
    return vocab


def build_sample(vocab, sentence_length):
    x = [random.choice([i for i in 'qwertyuiopsdfghjklzxcvbnm']) for _ in range(sentence_length-1)]
    y=random.randint(0,sentence_length-1)
    x.insert(y,'a')
    # print(x)
    x = [vocab.get(word, vocab['unk']) for word in x]   #将字转换成序号，为了做embedding
    return x, y


def build_dataset(sample_length, vocab, sentence_length):
    x = []
    y = []
    for i in range(sample_length):
        x_, y_ = build_sample(vocab, sentence_length)
        x.append(x_)
        y.append(y_)
    return torch.LongTensor(x), torch.LongTensor(y)


#测试代码 用来测试每轮模型的准确率
def evaluate(model, vocab, sample_length):
    model.eval()   # 表示预测
    x_test,y_test=build_dataset(200,vocab,sample_length)
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pre=model.forward(x_test)
        for i ,j in zip(y_test,y_pre):
            if i== j.argmax():
                correct+=1
            else:
                wrong+=1
    print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)


def main():
    #配置参数
    epoch_num = 40        #训练轮数
    batch_size = 20       #每次训练样本个数
    train_sample = 800    #每轮训练总共训练的样本总数
    char_dim = 10         #每个字的维度
    sentence_length = 6   #样本文本长度
    learning_rate = 0.005 #学习率
    # 建立字表
    vocab = build_vocab()
    # print(vocab)
    # print(vocab.keys())
    # 建立模型
    model = RnnTM(sentence_length, char_dim, vocab)
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
        print("=====================\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model, vocab, sentence_length)   #测试本轮模型结果
        log.append([acc, np.mean(watch_loss)])
    #画图
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  #画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  #画loss曲线
    plt.legend()
    plt.show()
    #保存模型
    # torch.save(model.state_dict(), "model.pth")
    # # 保存词表
    # writer = open("vocab.json", "w", encoding="utf8")
    # writer.write(json.dumps(vocab, ensure_ascii=False, indent=2))
    # writer.close()
    return

if __name__ == "__main__":
    main()