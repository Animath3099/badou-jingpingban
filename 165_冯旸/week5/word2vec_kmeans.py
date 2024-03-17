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

def get_key(dict, value):
    for key, val in dict.items():
        val = np.array(val)
        val = val.squeeze()
        if np.array_equal(val, value):
            return key



def main():
    model = load_word2vec_model("model.w2v") #加载词向量模型
    sentences = load_sentence("titles.txt")  #加载所有标题
    vectors = sentences_to_vectors(sentences, model)   #将所有标题向量化

    sen_vec = defaultdict(list)
    for sen, vec in zip(sentences, vectors):
        sen_vec[sen].append(vec)    #建立句子跟文本向量的关联


    n_clusters = int(math.sqrt(len(sentences)))  #指定聚类数量
    print("指定聚类数量：", n_clusters)
    kmeans = KMeans(n_clusters)  #定义一个kmeans计算类
    kmeans.fit(vectors)          #进行聚类计算

    # sentence_label_dict = defaultdict(list)
    # for sentence, label in zip(sentences, kmeans.labels_):  #取出句子和标签
    #     sentence_label_dict[label].append(sentence)         #同标签的放到一起

    vectors_label_dict = defaultdict(list)
    sentences_distance = defaultdict(list)
    for vectors, label in zip(vectors, kmeans.labels_):  # 取出向量和标签
        vectors_label_dict[label].append(vectors)
        array1 = vectors
        array2 = kmeans.cluster_centers_[label]
        # 计算余弦相似度
        cosine_similarity = np.dot(array1, array2) / (np.linalg.norm(array1) * np.linalg.norm(array2))
        # print("Cosine Similarity:", cosine_similarity)

        # 计算余弦距离（1-余弦相似度）
        distance = 1 - cosine_similarity
        # print("Cosine Distance:", distance)
        sen = get_key(sen_vec, vectors)
        sentences_distance[sen].append(distance)

    for label, vectors in vectors_label_dict.items():
        print("cluster %s :" % label)
        sen_dis = defaultdict(list)
        for vec in vectors:
            sen = get_key(sen_vec, vec)
            dis = sentences_distance[sen]
            sen_dis[sen].append(dis)
        sorted_sen_dis = sorted(sen_dis.items(), reverse=True, key=lambda x: x[1])
        sorted_values = [val for val, dic in sorted_sen_dis]
        for i in range(min(10, len(sorted_values))):
            print(sorted_values[i].replace(" ", ""))
        print("---------")






    # for label, sentences in sentence_label_dict.items():
    #     print("cluster %s :" % label)
    #     for i in range(min(10, len(sentences))):  #随便打印几个，太多了看不过来
    #         print(sentences[i].replace(" ", ""))
    #     print("---------")

if __name__ == "__main__":
    main()

