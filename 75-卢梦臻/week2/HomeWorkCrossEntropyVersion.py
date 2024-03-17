# _*_ coding: UTF-8 _*_
# @Time : 2024/1/16 21:08
# @Author : Yujez
# @File : HomeWorkCrossEntropyVersion
# @Project : intro_to_ml

import numpy as np
import torch
import torch.nn as nn
import torch.optim as op

# 1.创建单条数据
def create_single_sample():
    x=np.random.random(5)
    return x,np.argmax(x)
def create_samples(sample_nums):
    X=[]
    Y=[]

    for index in range(sample_nums):
        x,y=create_single_sample()
        X.append(x)
        Y.append(y)
    return torch.FloatTensor(X),torch.LongTensor(Y)

# 2.搭建网络层
class MaxModel(nn.Module):
    def __init__(self,input_size):
        super().__init__()
        self.layer1=nn.Linear(input_size,5)
        self.loss=nn.functional.cross_entropy
    def forward(self,x,y=None):
        y_pred=self.layer1(x)
        if y==None:
            return y_pred
        else:
            return self.loss(y_pred,y)

# 3.训练过程
def calculation():
    epoch_nums=50   # 训练轮数
    batch_size=40   # 每一轮中分成多少个批次
    samples_nums=5000   # 一轮总共训练5000个批次号
    lr=0.001
    model=MaxModel(5)
    optim=op.Adam(model.parameters(),lr)
    log=[]
    # train_x,train_y=create_samples(samples_nums)
    for epoch_num in range(epoch_nums):
        model.train()
        epoch_loss=[]
        for batch_index in range(samples_nums//batch_size):
            x,y=create_samples(batch_size)
            # x=train_x[batch_index*batch_size:(batch_index+1)*batch_size]
            # y=train_y[batch_index*batch_size:(batch_index+1)*batch_size]
            loss=model.forward(x,y)
            epoch_loss.append(loss.item())
            loss.backward()         # 权重更新
            optim.step()            #
            optim.zero_grad()
        acc=evaluate(model)
        log.append([acc,np.mean(epoch_loss)])
        print(f"当前轮：{epoch_num+1}，平均loss值：{np.mean(epoch_loss)}")
        torch.save(model.state_dict(),'./homework.pt')
    draw(log)

# 4.评判模型好坏
def evaluate(model):
    model.eval()
    sample_nums=100
    test_x,test_y=create_samples(sample_nums)
    correct,wrong=0,0
    with torch.no_grad():       # 不需要更新梯度
        y_pred=model.forward(test_x)
        for y_p,y_t in zip(y_pred,test_y):
            if np.argmax(y_p)==y_t:
                correct+=1
            else:
                wrong+=1
    print(f"此次验证正确率：{correct/(correct+wrong)}")
    return correct/(correct+wrong)
# 5.画图
import matplotlib.pyplot as plt
def draw(log):
    plt.plot(range(len(log)),[l[0] for l in log],label='acc')
    plt.plot(range(len(log)),[l[1] for l in log],label='loss')
    plt.legend()
    plt.show()

# 6.验证模型正确性
def predict(model_path,input_vec):
    model=MaxModel(5)
    model.load_state_dict(torch.load(model_path))  # 加载参数
    model.eval()
    with torch.no_grad():
        result=model.forward(torch.FloatTensor(input_vec))
    for vec,res in zip(input_vec,result):
        print(f"输入：{vec}，输出类别：{np.argmax(res)}，输出值：{res}")

# 7.入口
if __name__=='__main__':
    calculation()
    test_vec = [[0.47889086, 0.15229675, 0.31082123, 0.03504317, 0.18920843],
                [0.94963533, 0.5524256, 0.95758807, 0.95520434, 0.84890681],
                [0.78797868, 0.67482528, 0.13625847, 0.34675372, 0.09871392],
                [0.89349776, 0.59416669, 0.92579291, 0.41567412, 0.7358894]]
    predict('./homework.pt',test_vec)
