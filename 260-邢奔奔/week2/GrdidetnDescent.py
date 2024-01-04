#coding:utf8
import matplotlib.pyplot as plt 
import torch
import torch.nn as nn
import numpy as np 

input_data = [x * 0.01 for x in range(100)]
target_data = [2*x**2 + 3*x + 4 for x in input_data]
print(input_data)
print(target_data)
plt.scatter(input_data, target_data, color='red')
# plt.show()
#input()
def fun(x):
    y = w1 * x**2 + w2 *x + w3
    return y 
def loss(y_pred,y_true):
    return (y_pred - y_true)**2

lr = 0.1
w1, w2, w3 = -1, 0 ,1
epoch = 1000
batch_size = 100
for i in range(epoch):
    epoch_loss = 0 
    for x, y in zip(input_data,target_data):
        #clear grad
        gradw1, gradw2, gradw3 = 0, 0, 0
        for j in range(batch_size):
            y_pred =fun(x)
            epoch_loss += loss(y_pred,y)
            gradw1 += 2 * (y_pred -y ) * x**2 / batch_size
            gradw2 += 2 * (y_pred - y) * x / batch_size
            gradw3 += 2 * (y_pred - y) / batch_size

        w1 = w1 - lr * gradw1
        w2 = w2 - lr * gradw2
        w3 = w3 - lr * gradw3
    epoch_loss = np.mean(epoch_loss)
    print("%d轮的epcoh_loss %f"%(i+1, epoch_loss))
    if epoch_loss < 0.00001:
        break
print(f"训练后权重:w1:{w1} w2:{w2} w3:{w3}")
# print("训练后的权重为：{w1}, {w2}, {w3}".format(w1=w1,w2=w2,w3=w3))

y_pred = [fun(x) for x in input_data]
plt.scatter(input_data, target_data, color='red')
plt.scatter(input_data, y_pred)
plt.show()