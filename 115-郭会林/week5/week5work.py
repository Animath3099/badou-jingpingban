#!/usr/bin/env python3  
#coding: utf-8

#基于训练好的词向量模型进行聚类
#聚类采用Kmeans算法
import math
import re
import json
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
    kmeans = KMeans(n_clusters)  #定义一个kmeans计算类
    kmeans.fit(vectors)          #进行聚类计算

    vector_label_dict = defaultdict(list)

    # 计算平均距离
    # 选出距离最远的label并排除
    # 重新进行聚类计算

    for vector, label in zip(vectors, kmeans.labels_):  #取出句子向量和标签
        vector_label_dict[label].append(vector)         #同标签的放到一起

    dists = defaultdict(list)

    for label_vector, center in zip(sorted(vector_label_dict), kmeans.cluster_centers_): #取出标签和中心点
        dist = [np.linalg.norm(i - center) for i in vector_label_dict[label_vector]] # 计算每个样本与中心点欧几里得距离
        dists[label_vector].append(np.mean(dist)) # 计算平均距离

    sort_dist = sorted(dists.items(), key=lambda item: item[1], reverse=True) # 按照距离从小到大排序
    print(sort_dist)
    # new_centers = sort_dist[3:]  #这里想实现删掉远的点重新聚类的 但是没搞明白 后面再写
    # print(new_centers)

    # for label, sentences in sentence_label_dict.items():
    #     print("cluster %s :" % label)
    #     for i in range(min(10, len(sentences))):  #随便打印几个，太多了看不过来
    #         print(sentences[i].replace(" ", ""))
    #     print("---------")
    #     break


if __name__ == "__main__":
    main()

