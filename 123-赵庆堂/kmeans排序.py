import math
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as pyplot
# 生成一个包含100个词,每个词的维度是2的语料,设置为2是为了方便通过pyplot可视化进行查看
corpus = np.random.rand(100, 2)
# 初始化kmeans,将质点数量设置为语料中词总数的二次方根(经验质点数),需要注意的是math.sqrt结果是float,需要转换为int格式
# .fit_predict训练模型并预测结果,结果分类从0开始到分类数量-1,eg:分类数量为3,结果为0,1,2
my_kmeans = KMeans(n_clusters = int(math.sqrt(len(corpus)))).fit(corpus)
# 生成一个等于词表长度的列表(numpy.ndarray数字类型)用于存储每一个词属于哪个簇
label = my_kmeans.labels_
# print(label, type(label))
# 生成每个簇的质点,长度是簇的数量,生成数字类型为((numpy.ndarray)
center_point = my_kmeans.cluster_centers_
# print(center_point, type(center_point))
# print('0簇的点：：：',corpus[label == 0])
# print('质心：：：',center_point[0])
# print(np.linalg.norm(corpus[label == 0] - center_point[0], axis=1))


sort_clusters_result = list()
# 遍历所有的簇标签
for i in range(int(math.sqrt(len(corpus)))):
    # 取出对应标签簇的所有数据
    cluster_data = corpus[label == i]
    # 计算每个簇中每个点到质心的距离,注意选择轴
    distances = np.linalg.norm(cluster_data - center_point[i], axis=1)
    # 将每个簇中的每个点和每个点到质心的距离写入列表
    point_and_distance = list(zip(cluster_data, distances))
    sorted_points = sorted(point_and_distance, key=lambda x: x[1])
    # 将排序好的每个点取出
    sorted_cluster = [point_result for point_result, _ in sorted_points]
    sort_clusters_result.append(np.array(sorted_cluster))

# 打印一下每个簇的所有点的坐标,以及分类标签
for i, cluster in enumerate(sort_clusters_result):
    print('Cluster{}:'.format(i))
    print(cluster)

colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']  # 可用颜色列表
for i, cluster in enumerate(sort_clusters_result):
    # 如果簇的数量超过可用颜色的数量，循环使用颜色
    color = colors[i % len(colors)]

    # 绘制每个点到其质心的距离，点的大小根据距离缩放
    cluster_distances = [np.linalg.norm(point - center_point[i]) for point in cluster]
    sizes = [50 * (1 / dist) for dist in cluster_distances]  # 点的大小与距离成反比

    # 绘制散点图
    pyplot.scatter(cluster[:, 0], cluster[:, 1], c=color, s=sizes, alpha=0.6)

# 绘制质心
pyplot.scatter(center_point[:, 0], center_point[:, 1], c='b', s=200, alpha=1, marker='X')

# 显示图形
pyplot.title('KMeans Clustering Visualization')
pyplot.xlabel('Dimension 1')
pyplot.ylabel('Dimension 2')
pyplot.show()