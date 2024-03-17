import numpy as np
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# 加载模型
model = Word2Vec.load("Word_Vec4.model")

# 获取词向量
word_vectors = model.wv.vectors #从加载的模型中获取所有词向量，并存下来
words = model.wv.index_to_key #获取模型中所有词的列表，按照词频排序

# 应用 K-means 聚类
num_clusters = 5  #设定群组为5
kmeans = KMeans(n_clusters=num_clusters, random_state=0)  #设定随机状态为0
kmeans.fit(word_vectors)  #将词向量划分到5个组中
labels = kmeans.labels_ #将每个词向量的所组的标签存在label中，就是结果

# 使用 t-SNE 进行降维
tsne = TSNE(n_components=2, random_state=0)  #创建tsne 设定目标2维用于可视化，设定随机状态
reduced_vectors = tsne.fit_transform(word_vectors) #对向量进行降维并存储下来

# 可视化
plt.figure(figsize=(12, 8))  #创建一个12*8的图形
for i in range(num_clusters):
    # 提取每个聚类的数据并绘制
    idx = np.where(labels == i) #找到 每个i组的索引
    plt.scatter(reduced_vectors[idx, 0], reduced_vectors[idx, 1], label=f'Cluster {i}')
    # 画出x,y坐标 用不同颜色 label标记

plt.xlabel('TSNE Feature 1')
plt.ylabel('TSNE Feature 2')
plt.title('Word2Vec Vectors TSNE Visualization')
plt.legend()  #显示图例
plt.show()
