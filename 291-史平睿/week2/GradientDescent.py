import matplotlib.pyplot as pyplot
import math
import sys

X = [0.1*x for x in range(10)]
Y = [2*x**2 + 3*x + 4 for x in X]
#Y = [2*x**3 + 3*x**2 + 4*x for x in X]
#Z = [6.3787*x**2 + 2.4577*x + 0.1513 for x in X]
#print(X)
#print(Y)
#pyplot.scatter(X,Y, color='red')
#pyplot.scatter(X,Z)
#pyplot.show()

def func(x):
    return w1*x**2 + w2*x + w3

def loss(y_pred,y_true):
    return (y_pred-y_true)**2

w1,w2,w3 = -1,0,1
lr = -0.01
#lr = 0.2
batch_size = 5

for epochs in range(1000):
    epoch_loss=0
    grad_w1 = 0
    grad_w2 = 0
    grad_w3 = 0
    count = 0
    for x, y_true in zip(X,Y):
        y_pred = func(x)
        count += 1
        epoch_loss += loss(y_pred, y_true)

        grad_w1 += 2*(y_pred-y_true)*x**2
        grad_w2 += 2*(y_pred-y_true)*x
        grad_w3 += 2*(y_pred-y_true)

        if count == batch_size:
            w1 = w1 - lr*grad_w1/batch_size
            w2 = w2 - lr*grad_w2/batch_size
            w3 = w3 - lr*grad_w3/batch_size
            grad_w1 = 0
            grad_w2 = 0
            grad_w3 = 0
            count = 0

    epoch_loss /= len(X)
    print("第%d轮，loss %f" %(epochs, epoch_loss))
    if epoch_loss < 0.00001:
        break

print(f"训练后权重:w1:{w1} w2:{w2} w3:{w3}")

Yp = [func(i) for i in X]
#print(Yp)
pyplot.scatter(X,Y, color='red')
pyplot.scatter(X,Yp)
pyplot.show()