#!/usr/bin/env python3  
# coding: utf-8
import os

os.environ["OMP_NUM_THREADS"] = '1'
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


# 词向量，向量关系能反映语义关系，词义的相似性反映在向量的相似性
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


def main():
    model = load_word2vec_model("model.w2v")  # 加载词向量模型
    sentences = load_sentence("titles.txt")  # 加载所有标题
    vectors = sentences_to_vectors(sentences, model)  # 将所有标题向量化

    n_clusters = int(math.sqrt(len(sentences)))  # 指定聚类数量
    print("指定聚类数量：", n_clusters)
    kmeans = KMeans(n_clusters, n_init='auto')  # 定义一个kmeans计算类
    # 迭代聚类直到收敛
    previous_inertia = None
    while True:
        kmeans.fit(vectors)  # 进行聚类计算
        # 计算每个样本点到聚类中心的距离
        distances = kmeans.transform(vectors)

        # 找到离心距离最大的那个聚类中心
        max_distance_cluster = np.argmax(np.max(distances, axis=1))

        # 抛弃离心距离最大的那个聚类中心
        vectors = np.delete(vectors, max_distance_cluster, axis=0)
        sentences = list(sentences)
        del sentences[max_distance_cluster]

        if previous_inertia is not None and abs(previous_inertia - kmeans.inertia_) < 0.01:
            break
        previous_inertia = kmeans.inertia_

    kmeans.fit(vectors)  # 迭代到最后更新kmeans
    sentence_label_dict = defaultdict(list)
    for sentence, label in zip(sentences, kmeans.labels_):  # 取出句子和标签
        sentence_label_dict[label].append(sentence)  # 同标签的放到一起
    for label, sentences in sentence_label_dict.items():
        print("cluster %s :" % label)
        for i in range(min(10, len(sentences))):  # 随便打印几个，太多了看不过来
            print(sentences[i].replace(" ", ""))
        print("---------")


if __name__ == "__main__":
    main()
