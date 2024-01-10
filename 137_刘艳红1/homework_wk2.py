"""
作业：
基于pytorch框架编写模型训练
实现一个自行构造的找规律(机器学习)任务
规律：x是一个5维向量，如果第1个数最大，则返回0,第2个最大返回1，第3个最大返回2，第4个最大返回3，第5个最大返回4
"""

import numpy as np
import torch
import torch.nn as nn
from random import random
import matplotlib.pyplot as plt
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# nn.CrossEntropyLoss
# 构建一个函数生成样本数据
def sample_data():
    # y=[0]*5
    x=np.random.random(5)
    x_=x.tolist()
    y=x_.index(max(x_))
    return x,y


print(sample_data())
# 构建一个函数生成样本数据的个数
def sample(sample_num):
    X,Y=[],[]
    for i in range(sample_num):
        x,y=sample_data()
        X.append(x)
        Y.append(y)
    X,Y=np.array(X),np.array(Y)
    return torch.FloatTensor(X),torch.LongTensor(Y)

# 构造一个模型类
class TM(nn.Module):
    def __init__(self,input_size):
        super().__init__()
        self.lin=nn.Linear(input_size,5)
        self.act=torch.softmax
        self.loss=nn.functional.cross_entropy
    def forward(self,x,y):
        y_=self.lin(x)
        y_pre=self.act(y_,1)
        # print("检验预测值",y_pre.shape)
        # print("检验真实值", y.shape)
        loss=self.loss(y_pre,y)
        return loss
    def pred(self,x):
        y_ = self.lin(x)
        y_pre = self.act(y_,1)
        return y_pre

def pred_value(model):
    print("-开始预测")
    model.eval()
    x_train,y_train=sample(200)
    correct, wrong = 0, 0
    with torch.no_grad():
        y_preds=model.pred(x_train)
        # print(y_preds)
        for y_p, y_t in zip(y_preds, y_train):  # 与真实标签进行对比
            # print(y_p,y_t,"预测值与测试值")
            if y_p.argmax()==y_t:
                correct+=1
            else:
                wrong+=1
    print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)





def main():
    log = [] # 用来画图
    sample_num=9000
    x_test, y_test = sample(sample_num)
    epoch=30
    batch=30
    input_size=5
    model=TM(input_size)  # 模型实例化
    optim=torch.optim.Adam(model.parameters(),lr=0.05)  # 选择优化器并设置学习率
    for ep in range(epoch):
        print("======================================")
        print("-开始训练")
        model.train()
        loss_w=[]
        for ba in range(sample_num//batch):
            x=x_test[ba*batch:(ba+1)*batch]  # 切片取30组数据
            y=y_test[ba*batch:(ba+1)*batch]  # 切片取30组数据
            loss=model.forward(x,y)
            loss.backward()
            optim.step()
            optim.zero_grad()
            loss_w.append(loss)
        avg_loss=sum(loss_w)/len(loss_w)
        print(f'第{ep+1}轮平均loss为{avg_loss}')
        correct_rate=pred_value(model)
        # print(avg_loss,float(avg_loss))
        log.append([correct_rate, float(avg_loss)])
    # torch.save(model.state_dict(), "myhomewor01model.pt")      # 保存模型
    # print(log)

    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    plt.legend()
    plt.show()
    return


if __name__ == "__main__":
    main()
