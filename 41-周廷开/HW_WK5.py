#!/usr/bin/env python3
# coding: utf-8

import math
import jieba
import numpy as np
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from collections import defaultdict

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
        for word in words:
            try:
                vector += model.wv[word]
            except KeyError:
                vector += np.zeros(model.vector_size)
        vectors.append(vector / len(words))
    return np.array(vectors)


def compute_average_distances(vectors, labels, cluster_centers):
    distances = defaultdict(list)
    for vector, label in zip(vectors, labels):
        distance = np.linalg.norm(vector - cluster_centers[label])
        distances[label].append(distance)
    average_distances = {label: np.mean(distances[label]) for label in distances}
    return average_distances


def main():
    model = load_word2vec_model("model.w2v")  # 加载词向量模型
    sentences = load_sentence("titles.txt")  # 加载所有标题
    vectors = sentences_to_vectors(sentences, model)  # 将所有标题向量化

    n_clusters = int(math.sqrt(len(sentences)))  # 指定聚类数量
    print("指定聚类数量：", n_clusters)
    kmeans = KMeans(n_clusters)  # 定义一个kmeans计算类
    kmeans.fit(vectors)  # 进行聚类计算

    average_distances = compute_average_distances(vectors, kmeans.labels_, kmeans.cluster_centers_)
    # 按平均距离排序，获取排序后的类别标签
    sorted_labels = sorted(average_distances, key=average_distances.get)

    # 根据需要舍弃部分类别，这里假设要舍弃平均距离最长的10%的类别
    num_to_discard = int(0.1 * n_clusters)
    labels_to_discard = sorted_labels[-num_to_discard:]

    # 过滤掉要舍弃的类别中的句子
    sentence_label_dict = defaultdict(list)
    for sentence, label in zip(sentences, kmeans.labels_):
        if label not in labels_to_discard:
            sentence_label_dict[label].append(sentence)

    for label, sentences in sentence_label_dict.items():
        print("cluster %s :" % label)
        for i in range(min(10, len(sentences))):  # 随便打印几个，太多了看不过来
            print(sentences[i].replace(" ", ""))
        print("---------")


if __name__ == "__main__":
    main()