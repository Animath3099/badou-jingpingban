import math
from collections import defaultdict

import jieba
import numpy as np
from gensim.models import Word2Vec
from sklearn.cluster import KMeans


# 将每行句子转成向量形式进行表示
def transform_to_vector(corpus_path, vec_model):
    vectors = []
    # 将文本使用结巴分词，然后转成向量形式
    with  open(corpus_path, 'r', encoding='utf8') as corpus:
        # 循环每行句子
        for line in corpus:
            sentence = []
            # 将每行句子进行切词后进行循环
            for word in jieba.lcut(line):
                word = word.strip()
                if word != '' and word:
                    # 将切词转为向量
                    try:
                        # 没有get方法，vec_model.wv.get(word.strip())
                        sentence.append(vec_model.wv[word.strip()])
                    except KeyError:
                        # 部分词在训练中未出现，用全0向量代替
                        print(f'问题句子：{line}，不存在的单词:{word}\n')
            vectors.append(sentence)
    return vectors


import numpy as np
from scipy.spatial.distance import cdist

# Distance-Constrained Clustering Algorithm
# 基于距离的聚类算法第一版，先不使用KD树或者基于栅格的方法来优化性能。
#
# 1、先求出所有向量的两两距离
# 2、先取出距离最小的两个向量成为一组，如果有多个则形成多个组
# 3、从剩余的距离中，再次取两两距离最小的，如果取出来的元素，有存在于既有分组的，则另外一个元素也归于这个分组，直到分配完毕
#vectors 为所有数据，max_distance为最大距离
def DCC(vectors):
    # 计算所有向量之间的距离
    dists = cdist(vectors, vectors)

    # 进行聚类,距离小于阈值的向量放入同一聚类
    clusters = []
    for i in range(len(vectors)):
        grouped = False
        for cluster in clusters:
            if np.any(dists[i, cluster] < max_distance):
                cluster.append(i)
                grouped = True
                break
        if not grouped:
            clusters.append([i])

    print(clusters)



def main():
    # 加载词向量模型
    model = Word2Vec.load('model.w2v')
    # 将所有文本都转为向量
    vectors = transform_to_vector('titles.txt', model)

    n_clusters = int(math.sqrt(len(vectors)))  # 指定聚类数量
    print("指定聚类数量：", n_clusters)
    kmeans = KMeans(n_clusters)  # 定义一个kmeans计算类
    kmeans.fit(vectors)  # 进行聚类计算

    label_dict = defaultdict([])
    for word, label in kmeans.labels_:
        print(f'{word}: {label}')


if __name__ == '__main__':
    main()
