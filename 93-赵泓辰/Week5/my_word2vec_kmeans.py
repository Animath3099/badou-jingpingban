# !/usr/bin/env python3
# coding: utf-8

# 基于训练好的词向量模型进行聚类
# 聚类采用Kmeans算法
import math
import re
import json
import jieba
import numpy as np
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from collections import defaultdict
import statistics


# 输入模型文件路径
# 加载训练好的模型
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


# 将文本向量化
def sentences_to_vectors(sentences, model):
    vectors = []
    for sentence in sentences:
        words = sentence.split()  # sentence是分好词的，空格分开
        vector = np.zeros(model.vector_size)
        # 所有词的向量相加求平均，作为句子向量
        for word in words:
            try:
                vector += model.wv[word]
            except KeyError:
                # 部分词在训练中未出现，用全0向量代替
                vector += np.zeros(model.vector_size)
        vectors.append(vector / len(words))
    return np.array(vectors)


def calc_and_sort_distances(kmeans_model, X):
    distances = kmeans_model.transform(X)
    labels = kmeans_model.labels_
    cluster_distances = {}
    for i, label in enumerate(labels):
        if label not in cluster_distances:
            cluster_distances[label] = []
        cluster_distances[label].append(distances[i, labels])
    
    average_distance = {label: np.mean(distances) for label, distances in cluster_distances.items()}
    sorted_average_distance = dict(sorted(average_distance.items(), key=lambda item: item[1]))
    return sorted_average_distance


def main():
    model = load_word2vec_model("model.w2v") # 加载词向量模型
    sentences = load_sentence("titles.txt")  # 加载所有标题
    vectors = sentences_to_vectors(sentences, model)   # 将所有标题向量化

    n_clusters = int(math.sqrt(len(sentences)))  # 指定聚类数量
    print("指定聚类数量：", n_clusters)
    kmeans = KMeans(n_clusters, random_state=42)  # 定义一个kmeans计算类
    kmeans.fit(vectors)          # 进行聚类计算

    centors = kmeans.cluster_centers_  # 获取第一次聚类质心
    sorted_average_distance = calc_and_sort_distances(kmeans, vectors)  # 将第一次聚类的结果按照类内距离排序
    new_n_clusters = 20  # 制定第二次聚类数量
    new_centors = centors[list(sorted_average_distance.keys())[:new_n_clusters]]  # 获取第二次聚类质心初始位置
    print("指定新的聚类数量：", new_n_clusters)
    new_kmeans = KMeans(new_n_clusters, init=new_centors, random_state=42)  # 制定质心初始位置并聚类
    new_kmeans.fit(vectors)

    sentence_label_dict = defaultdict(list)
    for sentence, label in zip(sentences, new_kmeans.labels_):  # 取出句子和标签
        sentence_label_dict[label].append(sentence)         # 同标签的放到一起
    for label, sentences in sentence_label_dict.items():
        print("cluster %s :" % label)
        for i in range(min(10, len(sentences))):  # 随便打印几个，太多了看不过来
            print(sentences[i].replace(" ", ""))
        print("---------")

if __name__ == "__main__":
    main()

