import random
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.nn as nn
import numpy as np
import json
import matplotlib.pyplot as plt



def build_vacob():
    alpha = 'abcdefghijklmnopqrstuvwxyz'
    vacob = {'pad': 0}

    for index, a in enumerate(alpha):
        vacob[a] = index + 1

    vacob['unk'] = len(vacob)
    return vacob


def build_sentence(vacob, sentence_length, letter='a'):
    sentence = [vacob[random.choice('bcdefghijklmnopqrstuvwxyz')] for _ in range(sentence_length)]

    if vacob[letter] not in sentence:
        # sentence[random.randint(0, sentence_length - 1)] = vacob[letter]
        if random.random() < 0.2:
            # print('-------11111---------')
            return sentence,sentence_length # 30
        else:
            # print('-------22222---------')
            sentence[random.randint(0, sentence_length - 1)] = vacob[letter]
            return sentence,sentence.index(vacob[letter])
    else:

        return sentence, sentence.index(vacob[letter]) # 0 - 29


def build_dataset(vacob, batch_size, sentence_length):
    sentences = []
    y = []

    for _ in range(batch_size):
        sentence, index = build_sentence(vacob, sentence_length)
        sentences.append(sentence)
        y.append(index)

    return torch.LongTensor(sentences), torch.LongTensor(y)


hidden_size = 12
linear_output_size = 7


class NlpSimpleModule(nn.Module):
    def __init__(self, letter_dim, vacob):
        super().__init__()

        self.embeding = nn.Embedding(len(vacob), letter_dim)
        self.rnn_pooling = nn.RNN(letter_dim, hidden_size,bias=False, batch_first=True)
        self.loss = nn.functional.cross_entropy

    def forward(self, x, y=None):
        x1 = self.embeding(x)
        _, x2 = self.rnn_pooling(x1)

        y_pred = x2.squeeze()
        # print('yp:',np.shape(y_pred))
        if y is not None:
            # print('y:',np.shape((y)))
            return self.loss(y_pred, y)
        else:
            return y_pred


def evaluate(model,vocab,batch_size,sentence_len):
    model.eval()
    x,y = build_dataset(vocab,batch_size,sentence_len)
    #
    # y_unique = dict(zip(*np.unique(y, return_counts=True)))
    #
    # print('本次预测集中共有%d个0样本，%d个1样本，%d个2样本，%d个3样本，%d个4样本，%d个5样本,%d个5样本'
    #       %
    #       (y_unique[0], y_unique[1], y_unique[2], y_unique[3], y_unique[4],y_unique[5],y_unique[6]))


    correct,wrong = 0,0

    with torch.no_grad():
        y_pred = model(x)


        for y_p,y_t ,x_t in zip(y_pred,y,x):
            # print(y_p,'-->',y_t,'-->',x_t)
            if np.argmax(y_p) == y_t:
                correct+=1
            else:
                wrong+=1

        print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
        return float(correct / (correct + wrong))


def main_train():
    # 配置参数
    train_times = 20
    batch_size = 20
    train_sample_nums = 5000
    letter_dim = 10
    sentence_len = 9
    learning_rate = 0.005

    vacob = build_vacob()

    my_module = NlpSimpleModule(letter_dim, vacob)

    optim = torch.optim.Adam(my_module.parameters(), lr=learning_rate)

    logs = []

    for train_time in range(train_times):

        my_module.train()

        watch_loss = []

        for batch in range(int(train_sample_nums/batch_size)):
            x,y = build_dataset(vacob,batch_size,sentence_len)

            optim.zero_grad()

            loss = my_module.forward(x,y)

            loss.backward()

            optim.step()

            watch_loss.append(loss.item())

        print('==================='
              '\n第%d轮平均loss:%f'
              %
              (train_time+1,np.mean(watch_loss)))

        acc = evaluate(my_module,vacob,200,sentence_len)
        print('acc:',acc)
        logs.append([acc,np.mean(watch_loss)])

    print(logs)
    #画图
    plt.plot(range(len(logs)),[l[0] for l in logs],label = 'acc')
    plt.plot(range(len(logs)),[l[1] for l in logs],label = 'loss')
    plt.legend()
    plt.show()

    torch.save(my_module.state_dict(),'nlp_module1.pth')

    return


if __name__ == '__main__':

    main_train()
