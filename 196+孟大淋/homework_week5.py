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
    # print(vectors, vectors.shape)
    n_clusters = int(math.sqrt(len(sentences)))  #指定聚类数量
    print("指定聚类数量：", n_clusters)
    kmeans = KMeans(n_clusters)  #定义一个kmeans计算类
    kmeans.fit(vectors)          #进行聚类计算

    sentence_label_dict = defaultdict(list)
    vector_label_dict = defaultdict(list)
    # print(sentence_label_dict)
    # print('kmeans.labels_, kmeans.cluster_centers_ = ', kmeans.labels_, kmeans.cluster_centers_.shape, len(kmeans.labels_))
    for sentence, vector, label in zip(sentences, vectors, kmeans.labels_):  #取出句子和标签   还可以用kmeans.claster_center来提取中心
        # print(len(sentence), vector.shape, label)
        # print('label, distance = ',label, distances)
        sentence_label_dict[label].append(sentence)         #同标签的放到一起
        vector_label_dict[label].append(vector)
    # print(sentence_label_dict)
    j=0
    distance_sort = defaultdict(list)
    for label, sentences in sentence_label_dict.items():
        vector_for_dis = vector_label_dict[label]
        distances = []
        center = kmeans.cluster_centers_

        for i in range(len(sentences)):
            distance = np.linalg.norm(center[j] - vector_for_dis[i])
            # print('distance=', distance)
            # print('第%d次', i)
            distances.append(distance)

        j += 1

        distance_sort[label+1].append(sum(distances) / len(distances))

        print("cluster %s ，共有%d条句子 :" % (label+1, len(sentences)))
        print("类内平均距离 = %f" %(sum(distances) / len(distances)))
        print("=====句子如下======")

        for i in range(min(10, len(sentences))):  #随便打印几个，太多了看不过来
            print(sentences[i].replace(" ", ""))
        print("三三三三三三三三三三三三三三三三三三三三三三三三三三三三三三三三三三三三三三三三三三三三三三三")

    # print(distance_sort)
    # 按类内距离从小到大排序，最小的说明分类最好
    paixu = list(zip(distance_sort.values(), distance_sort.keys()))
    paixu = sorted(paixu, reverse=False)
    print("排序情况如下：")
    print(paixu)

if __name__ == "__main__":
    main()

