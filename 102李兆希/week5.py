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
            #Python strip() 方法用于去除字符串开头和结尾处指定的字符(默认为空格或换行符)或字符序列，不会去除字符串中间对应的字符。
            sentences.add(" ".join(jieba.cut(sentence)))
    print("获取句子数量：", len(sentences))
    return sentences

#将文本向量化
def sentences_to_vectors(sentences, model):
    vectors = []#有多少行句子vectors就有多少个向量 1xn?
    for sentence in sentences:
        words = sentence.split()  #sentence是分好词的，空格分开 上面jieba分了词的
        vector = np.zeros(model.vector_size)
        #所有词的向量相加求平均，作为句子向量
        for word in words:
            try:#发现异常就报错 try后面跟的执行的代码 except后面跟的发生异常时执行的代码
                vector += model.wv[word]
            except KeyError:
                #部分词在训练中未出现，用全0向量代替
                vector += np.zeros(model.vector_size)
        vectors.append(vector / len(words))
    return np.array(vectors)


def main():
    model = load_word2vec_model("model.w2v") #加载词向量模型
    sentences = load_sentence("titles.txt")  #加载所有标题
    #print(sentences)
    vectors = sentences_to_vectors(sentences, model)   #将所有标题向量化
    #print(vectors.shape)#(1796, 100)
    n_clusters = int(math.sqrt(len(sentences)))  #指定聚类数量
    print("指定聚类数量：", n_clusters)
    kmeans = KMeans(n_clusters)  #定义一个kmeans计算类
    #print(kmeans)

    kmeans.fit(vectors)          #进行聚类计算
    #print(kmeans.fit(vectors))
    centers = kmeans.cluster_centers_  # 42组数据点的中心点 #print(centers.shape)#(42, 100)
    labels = kmeans.labels_  # 每个数据点所属分组
    #print(labels)
    # print(vectors.shape)#(1796, 100)
    #---------------------------------------
    # 把下面每一类的平均距离算出来
    #实现类内距离排序
    i=0
    vector_label_dict = defaultdict(list)
    for label in labels:#label从0到41
        vector_label_dict[label].append(vectors[i])
        i+=1
    for label, vectors in vector_label_dict.items():
        #total_distance,distance=0,[]
        total_distance = 0
        distance=list()
        print("cluster %s :" % label)
        for i in range(len(vectors)):
            total_distance+=np.linalg.norm(vectors[i]-centers[label])
            distance.append(np.linalg.norm(vectors[i]-centers[label]))

            avg_distance=total_distance/len(vectors)
        distance.sort()
        print("类内每个向量到中心点的距离：",distance)
        #print(vectors.__sizeof__())
        print("平均距离:%f" % avg_distance)
#---------------------------
        #print("---------")


    #for
    #print(vector_label_dict.shape)
    # arr = np.array(vector_label_dict)
    # print(arr.shape)
        #print(label)
        #对1796个label 计算

    #print(labels.shape)#(1796,)
    #print(labels)

    # sentence_label_dict = defaultdict(list)
    # for sentence, label in zip(sentences, kmeans.labels_):  #取出句子和标签
    #     #zip 用法
    #     # >> > a = ['a', 'b', 'c', 'd']
    #     # >> > b = ['1', '2', '3', '4']
    #     # >> > list(zip(a, b))
    #     # [('a', '1'), ('b', '2'), ('c', '3'), ('d', '4')]
    #
    #     sentence_label_dict[label].append(sentence)         #同标签的放到一起
    #print(sentence_label_dict[0])
    # for label, sentences in sentence_label_dict.items():
    #     print("cluster %s :" % label)
    #     for i in range(min(10, len(sentences))):  #随便打印几个，太多了看不过来
    #          print(sentences[i].replace(" ", ""))
    #     print("---------")

if __name__ == "__main__":
    main()

