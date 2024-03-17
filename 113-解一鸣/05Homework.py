#!/usr/bin/env python3  
#coding: utf-8

#基于训练好的词向量模型进行聚类
#聚类采用Kmeans算法
import math
import jieba
import numpy as np
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from collections import defaultdict

#输入模型文件路径
#加载训练好的模型
def load_word2vec_model(path):
    model = Word2Vec.load(path)
    return model

def load_sentence(path):
    sentences = set()
    with open(path, encoding="utf8") as f:
        for line in f:
            sentence = line.strip()
            sentences.add(" ".join(jieba.cut(sentence)))
    print("获取句子数量：", len(sentences))
    return sentences

#将文本向量化
def sentences_to_vectors(sentences, model):
    vectors = []
    for sentence in sentences:
        words = sentence.split()  #sentence是分好词的，空格分开
        vector = np.zeros(model.vector_size)
        #所有词的向量相加求平均，作为句子向量
        for word in words:
            try:
                vector += model.wv[word]
            except KeyError:
                #部分词在训练中未出现，用全0向量代替
                vector += np.zeros(model.vector_size)
        vectors.append(vector / len(words))
    return np.array(vectors)


def main():
    model = load_word2vec_model("model.w2v") #加载词向量模型
    sentences = load_sentence("titles.txt")  #加载所有标题
    vectors = sentences_to_vectors(sentences, model)   #将所有标题向量化

    n_clusters = int(math.sqrt(len(sentences)))  #指定聚类数量
    print("指定聚类数量：", n_clusters)
    kmeans = KMeans(n_clusters, n_init='auto')  #定义一个kmeans计算类
    kmeans.fit(vectors)          #进行聚类计算
    
    cluster_centers = kmeans.cluster_centers_
    distances = []
    for i in range(n_clusters):
        cluster_points = [sentence for sentence, label in zip(sentences, kmeans.labels_) if label == i]
        if len(cluster_points) > 1:
            cluster_points_vectors = sentences_to_vectors(cluster_points, model)
            avg_distance = np.mean(np.linalg.norm(cluster_points_vectors - cluster_centers[i], axis=1))
            distances.append((i, avg_distance))

    distances.sort(key=lambda x: x[1], reverse=True)

    
    # 逐步舍弃类别
    selected_clusters = []
    for cluster, _ in distances:
        # 可以根据需要设置舍弃的条件，比如基于阈值
        if len(selected_clusters) / n_clusters < 0.8:
            selected_clusters.append(cluster)
        else:
            break

    selected_data_by_cluster = defaultdict(list)

    for sentence, label in zip(sentences, kmeans.labels_):
        if label in selected_clusters:
            selected_data_by_cluster[label].append(sentence)

    # 输出每个聚类中的句子
    for cluster, data in selected_data_by_cluster.items():
        print(f"聚类 {cluster}:")
        for sentence in data:
            print(sentence)
        print("---------")


if __name__ == "__main__":
    main()

