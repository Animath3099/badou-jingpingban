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

# 获取列表的第二个元素
def takeSecond(elem):
    return elem[1]


def main():
    model = load_word2vec_model("model.w2v") #加载词向量模型
    sentences = load_sentence("titles.txt")  #加载所有标题
    vectors = sentences_to_vectors(sentences, model)   #将所有标题向量化

    n_clusters = int(math.sqrt(len(sentences)))  #指定聚类数量
    print("指定聚类数量：", n_clusters)
    kmeans = KMeans(n_clusters)  #定义一个kmeans计算类
    kmeans.fit(vectors)          #进行聚类计算

    sentence_label_dict = defaultdict(list)
    for sentence, label in zip(sentences, kmeans.labels_):  #取出句子和标签
        sentence_label_dict[label].append(sentence)         #同标签的放到一起

    range_cluster = [0]*n_clusters      # 用列表储存每个簇的平均距离
    for label, sentences in sentence_label_dict.items():
        # 计算欧式距离的平均值，并存入range_cluster中
        cluster_vec = np.array(sentences_to_vectors(sentences, model))      # 加载簇中句子的句向量
        cluster_center = np.array(kmeans.cluster_centers_[label])           # 加载簇中心点
        distances = np.linalg.norm(cluster_vec - cluster_center, axis=1)    # 计算每个点到中心点的欧式距离
        range_cluster[label] = np.mean(distances)   # 将平均距离存储在列表当中
        # print("cluster %s :" % label)
        # for i in range(min(10, len(sentences))):  #随便打印几个，太多了看不过来
        #     print(sentences[i].replace(" ", ""))
        # print("---------")
    
    # 用列表记录各簇的平均距离的排序
    range_label = []
    for i in range(len(range_cluster)):
        range_label.append([i, range_cluster[i]])
    range_label.sort(key=takeSecond)
    # print(range_cluster)
    # print(range_label)
    print("类与平均距离对应表（已排序）：\n")
    print(range_label)
    # 打印平均欧式距离从小到大的前3的类
    print("\n\n平均欧式距离从小到大的前3的类：\n")
    for i in range(3):
        label = range_label[i][0]
        sentences = sentence_label_dict[label]
        print("cluster %s , 距离次序（从近到远） %d:" % (label,i+1))
        for i in range(min(10, len(sentences))):  #随便打印几个，太多了看不过来
            print(sentences[i].replace(" ", ""))
        print("---------")

    # 打印平均欧式距离从小到大的后3的类
    print("\n\n平均欧式距离从小到大的后3的类：\n")
    for i in range(3):
        label = range_label[i+39][0]
        sentences = sentence_label_dict[label]
        print("cluster %s , 距离次序（从近到远） %d:" % (label,i+40))
        for i in range(min(10, len(sentences))):  #随便打印几个，太多了看不过来
            print(sentences[i].replace(" ", ""))
        print("---------")

if __name__ == "__main__":
    main()

