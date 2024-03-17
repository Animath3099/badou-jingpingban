#!/usr/bin/env python3
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
    kmeans = KMeans(n_clusters)  # 定义一个kmeans计算类
    kmeans.fit(vectors)  # 进行聚类计算

    sentence_label_dict = defaultdict(list)
    vector_label_dict = defaultdict(list)
    vector_avg_distance_label_dict = dict()
    for sentence, label, vec in zip(sentences, kmeans.labels_, vectors):  # 取出句子和标签
        sentence_label_dict[label].append(sentence)  # 同标签的放到一起
        vector_label_dict[label].append(vec)

    vector_label_list = sorted(vector_label_dict.items(), key=lambda x: x[0])
    for vector_label, center_vec in zip(vector_label_list, kmeans.cluster_centers_):
        vec_distance_sum = 0
        for vec in vector_label[1]:
            # vec_distance = math.sqrt(sum([(x - y) ** 2 for x, y in zip(vec, center_vec)]))  # 欧式距离
            vec_distance = np.dot(vec, center_vec) / (np.linalg.norm(vec) * np.linalg.norm(center_vec))  # 余玄距离
            vec_distance_sum += vec_distance
        avg_distance = vec_distance_sum / len(vector_label[1])
        vector_avg_distance_label_dict[vector_label[0]] = avg_distance
    vector_avg_distance_label_dict_list = sorted(vector_avg_distance_label_dict.items(), key=lambda x: x[1],
                                                 reverse=True)

    vector_avg_distance_label_dict_list = vector_avg_distance_label_dict_list[:5]
    for label_avg_distance in vector_avg_distance_label_dict_list:
        print("cluster %s avg_distance %s:" % (label_avg_distance[0], label_avg_distance[1]))
        for sentence in sentence_label_dict.get(label_avg_distance[0]):
            print(sentence.replace(" ", ""))
        print("---------")


if __name__ == "__main__":
    main()
