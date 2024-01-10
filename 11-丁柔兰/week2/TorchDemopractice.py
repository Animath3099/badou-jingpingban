import matplotlib.pyplot as pyplot
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 生成多分类数据
np.random.seed(0)
X1 = np.random.rand(100, 2) + np.array([0.5, 0.5])
X2 = np.random.rand(100, 2) + np.array([1.5, 1.5])
X3 = np.random.rand(100, 2) + np.array([2.5, 2.5])
X = np.concatenate([X1, X2, X3])
y = np.array([0] * 100 + [1] * 100 + [2] * 100)

# 数据集分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 初始化并训练Logistic回归模型
model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
model.fit(X_train, y_train)

# 使用测试集进行预测
y_pred = model.predict(X_test)

# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print("准确率：", accuracy)

# 可视化分类结果
colors = ['r', 'g', 'b']
for i in range(3):
    pyplot.scatter(X_test[y_test == i][:, 0], X_test[y_test == i][:, 1], c=colors[i], label=str(i))
pyplot.legend(['Class 0', 'Class 1', 'Class 2'])
pyplot.show()