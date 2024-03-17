#Week5 Homework
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

    sentence_label_dict = defaultdict(list)
    label_center_dict = defaultdict(list)
    new_vectors_dict = defaultdict(list)
    for vectors, label in zip(sentences, kmeans.labels_):  #取出句子和标签
        sentence_label_dict[label].append(vectors)         #同标签的放到一起

    for label, cluster_center in zip(kmeans.labels_, kmeans.cluster_centers_): #取出标签和聚类中心
        label_center_dict[label].append(cluster_center) #标签&聚类中心

    for label, vector in label_center_dict.items(): #更新vector
        print("cluster %s :" % label)
        for i in range(len(vector)): 
            tmp = 0
            cluster_center = label_center_dict[label]
            tmp = pow(vector[i] - cluster_center,2)
            dis = pow(tmp,0.5)
            if dis.any() < 3:
                new_vectors_dict[label].append(vector)
            print(new_vectors_dict[label])
    
    for label, vectors in new_vectors_dict.items(): 
        print("Vectors for cluster %s :" % label)
        for vector in vectors:
            sentence_vectors = sentences_to_vectors(list(sentence_label_dict[label]), model)  # Convert the set to a list
            for sentence_vector in sentence_vectors:
                tmp = sum((a - b) ** 2 for a, b in zip(vector, sentence_vector))  # Calculate the Euclidean distance between vectors
                dis = pow(tmp, 0.5)
                if dis.any() < 3:
                    index = np.where(sentence_vectors == sentence_vector)[0][0]  # Get the index of the sentence vector in the array
                    index = list(sentence_label_dict[label])[index]  # Convert the index to a list and get the corresponding sentence
                    print(sentences[index])  # Print the corresponding sentence

if __name__ == "__main__":
    main()