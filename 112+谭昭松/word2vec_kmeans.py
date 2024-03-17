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
from sklearn.metrics.pairwise import cosine_similarity

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

    sentence_label_dict = defaultdict(list)
    cluster_O_distances = defaultdict(list)
    cluster_cosine_distances = defaultdict(list)

    for sentence, label in zip(sentences, kmeans.labels_):  #取出句子和标签
        sentence_label_dict[label].append(sentence)         #同标签的放到一起
        # 类内向量距离中心的欧式距离计算
        cluster_O_distances[label].append(np.linalg.norm(vectors[label] - kmeans.cluster_centers_[label]))
    

    # for label, sentences in sentence_label_dict.items():
    #     print("cluster %s :" % label)
    #     for i in range(min(10, len(sentences))):  #随便打印几个，太多了看不过来
    #         print(sentences[i].replace(" ", ""))


    #     # 计算余弦距离
    #     cluster_center = kmeans.cluster_centers_[label].reshape(1, -1)
    #     similarities = cosine_similarity(vectors[kmeans.labels_ == label], cluster_center)
    #     cluster_cosine_distances[label] = similarities.flatten()
    #     # 平均距离
    #     avg_O_distance = np.mean(cluster_O_distances[label])
    #     avg_cosine_distances = np.mean(cluster_cosine_distances[label])
        
    #     print("簇内句子向量与簇中心的平均欧氏距离：%s: %.4f" % (label, avg_O_distance))
    #     print("簇内句子向量与簇中心的平均余弦距离：%s: %.4f" % (label, avg_cosine_distances))
    #     print("---------")
    
    # 计算每个簇的平均欧氏距离和平均余弦相似度
    cluster_avg_distances = []
    for label, distances in cluster_O_distances.items():
        avg_O_distance = np.mean(distances)
        avg_cosine_similarity = np.mean(cosine_similarity(vectors[kmeans.labels_ == label], [kmeans.cluster_centers_[label]]))
        cluster_avg_distances.append((label, avg_O_distance, avg_cosine_similarity))

        # 按照平均欧氏距离排序
        sorted_clusters = sorted(cluster_avg_distances, key=lambda x: x[1])
        
        threshold = 0.9 # 保存百分之九十的类
        keep = int(threshold * len(sorted_clusters)) # 保存的类数
        clusters_keep = sorted_clusters[:keep]

    for label, avg_O_distance, avg_cosine_similarity in clusters_keep:
        print("保留的簇 %s :" % label)
        print("平均欧氏距离：%s: %.4f" % (label, avg_O_distance))
        print("平均余弦距离：%s: %.4f" % (label, avg_cosine_similarity))
        for sentence in sentence_label_dict[label][:min(10, len(sentence_label_dict[label]))]:
            print(sentence.replace(" ", ""))
        print("---------")


if __name__ == "__main__":
    main()

