# coding:utf8
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt

'''
基于pytorch框架编写模型训练
实现一个自行构造的找规律(机器学习)任务
规律：x是一个5维向量，最大的元素坐标为
0  为 0样本
1  为 1样本
2  为 2样本
3  为 3样本
4  为 4样本

[6,2,3,4,5] 0
[1,6,3,4,5] 1
[1,2,6,3,5] 2
[1,2,4,6,5] 3
[1,4,3,5,6] 4
'''




class TorchModel(nn.Module):
    def __init__(self,input_size):
        super(TorchModel, self).__init__()
        self.linear1 = nn.Linear(input_size,5)
        # self.linear2 = nn.Linear(3,5)
        # self.linear3 = nn.Linear(8,5)
        # self.activation = torch.sigmoid
        self.loss = nn.functional.cross_entropy

    def forward(self,x,y=None):
        y_pred = self.linear1(x)
        # x2 = self.linear2(x1)
        # x3 = self.linear3(x2)
        # y_pred = self.activation(x1)
        # print('预测值00：',y_pred[0][0])
        # if y_pred[0][0] >= 1:
        #
        #     print('预测值：',y_pred)
        print('y_pred size:', np.shape(y_pred))
        print('y size:', np.shape(y))
        input()
        if y is not None:
            # print('y_p-->',y_pred[0])
            # temp_y = torch.softmax(y_pred[0],-1)
            # print(temp_y)
            # input()
            return self.loss(y_pred,y)
        else:
            return y_pred

# 构建数据
def build_samples():
    x = np.random.random(5)
    return x,np.argmax(x)

# 构建数据集
def build_dataset(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        x,y = build_samples()
        X.append(x)
        Y.append(y)

    return torch.FloatTensor(np.array(X)),torch.LongTensor(np.array(Y))


def evaluate(model):
    model.eval()
    test_sample_num = 100
    x, y = build_dataset(test_sample_num)

    y_unique = dict(zip(*np.unique(y, return_counts=True)))

    print('本次预测集中共有%d个0样本，%d个1样本，%d个2样本，%d个3样本，%d个4样本'
          %
          (y_unique[0], y_unique[1], y_unique[2], y_unique[3], y_unique[4]))

    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)
        for y_p, y_t in zip(y_pred, y):

            if np.argmax(y_p) == y_t:
                correct += 1
            else:
                wrong += 1

        print('正确预测个数：%d,正确率：%f' % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)


# 使用训练好的模型做预测
def predict(model_path,input_vec):
    input_size =5
    model = TorchModel(input_size)
    model.load_state_dict(torch.load(model_path))
    print(model.state_dict())

    model.eval()
    with torch.no_grad():
        result = model.forward(torch.FloatTensor(input_vec))

    for vec,res in zip(input_vec,result):
        print(vec,'-->',np.argmax(np.array(res)),'-->',res)

def main():
    #配置参数
    epoch_num = 100 #训练轮数
    batch_size = 25 #每次训练样本个数
    train_sample = 10000 # 每轮训练的样本总数
    input_size = 5 # 输入的向量维度
    learning_rate = 0.001 #学习率

    #建立模型
    model = TorchModel(input_size)

    #选择优化器
    optim = torch.optim.Adam(model.parameters(),lr = learning_rate)

    log = []

    #创建训练集，正常任务是读取训练集
    train_x,train_y = build_dataset(train_sample)

    #训练过程
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []

        for batch_index in range(train_sample// batch_size):
            x = train_x[batch_index*batch_size:(batch_index+1)*batch_size]
            y = train_y[batch_index*batch_size:(batch_index+1)*batch_size]

            loss = model(x,y) #计算loss
            loss.backward() #计算梯度

            optim.step() #更新梯度
            optim.zero_grad() #梯度归零


            watch_loss.append(loss.item())

        acc = evaluate(model)
        log.append([acc,float(np.mean(watch_loss))])
        print('======第%d轮平均loss：%f'%(epoch+1,np.mean(watch_loss)))

    torch.save(model.state_dict(),'model.pt')

    print(log)
    plt.plot(range(len(log)),[l[0] for l in log],label='acc')#正确率
    plt.plot(range(len(log)),[l[1] for l in log],label='loss')#loss
    plt.legend()
    plt.show()
    return





if __name__ == '__main__':
    main()
    test_vec,y= build_dataset(10)
    predict("model.pt",test_vec)
