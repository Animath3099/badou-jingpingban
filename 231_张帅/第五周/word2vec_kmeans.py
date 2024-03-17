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


def distance_o(point1:np.ndarray, point2:np.ndarray):
    assert point1.shape == point2.shape
    distance = 0.0
    for i in range(len(point1)):
        distance += (point1[i] - point2[i]) ** 2
    return np.sqrt(distance)

def main():
    model = load_word2vec_model("model.w2v") #加载词向量模型
    sentences = load_sentence("titles.txt")  #加载所有标题
    vectors = sentences_to_vectors(sentences, model)   #将所有标题向量化

    n_clusters = int(math.sqrt(len(sentences)))  #指定聚类数量
    print("指定聚类数量：", n_clusters)
    kmeans = KMeans(n_clusters)  #定义一个kmeans计算类
    kmeans.fit(vectors)          #进行聚类计算

    sentence_label_dict = defaultdict(dict) # 例如：0 :{ "sentences":[...] , "sum_distance" : 20, "num" : 100  }

    labels = kmeans.labels_
    centers = kmeans.cluster_centers_
    for index, sentence in enumerate(sentences):
        vector = vectors[index]
        label = labels[index]
        center = centers[label]
        distance = distance_o(vector,center)

        if sentence_label_dict[label].get("sentences") is None:
            sentence_label_dict[label]["sentences"] = []
        if sentence_label_dict[label].get("sum_distance") is None:
            sentence_label_dict[label]["sum_distance"] = 0
        if sentence_label_dict[label].get("num") is None:
            sentence_label_dict[label]["num"] = 0

        sentence_label_dict[label]["sentences"].append(sentence)
        sentence_label_dict[label]["sum_distance"] += distance
        sentence_label_dict[label]["num"] += 1

    sentence_label_sorted = sorted(sentence_label_dict.items(),key=lambda x:x[1]["sum_distance"]/x[1]["num"])[:5]

    for label, sentence_dict in sentence_label_sorted:
        print("cluster %s :" % label)
        avg_distance = sentence_dict["sum_distance"] / sentence_dict["num"]
        print(f"avg_distance:{avg_distance}")
        sentences = sentence_dict["sentences"]
        for i in range(min(10, len(sentences))):  #随便打印几个，太多了看不过来
            print(sentences[i].replace(" ", ""))
        print("---------")

if __name__ == "__main__":
    main()

