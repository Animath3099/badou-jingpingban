
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
def load_model(path):
    model = Word2Vec.load(path)
    return model

def load_data(path):
    data = set()
    with open(path, encoding="utf8") as f:
        for line in f:
            sentence = line.strip()
            data.add(" ".join(jieba.cut(sentence)))
    print("获取数据数量：", len(data))
    return data


#将文本向量化
def sentences_to_vectors(data, model):
    vectors = []
    for sentence in data:#sentence是分好词的，空格分开
        words = sentence.split()
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

def cosine_distance(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def main():
    model = load_model("model.w2v") #加载词向量模型
    data = load_data("titles.txt") #加载所有标题
    vectors = sentences_to_vectors(data, model) #将所有标题向量化

    n_clusters = int(math.sqrt(len(data))) #指定聚类数量
    print("指定聚类数量：", n_clusters)
    kmeans = KMeans(n_clusters) #定义一个kmeans计算类
    kmeans.fit(vectors) #进行聚类计算

    grouped_data = defaultdict(list)
    for sentence, label in zip(data, kmeans.labels_): #取出句子和标签
        grouped_data[label].append(sentence) #同标签的放到一起

    # week5 作业 start
    cluster_centers = kmeans.cluster_centers_
    average_cos_distance = []
    for cluster_label, sentences in grouped_data.items():
        sentence_vectors = sentences_to_vectors(sentences, model)
        current_cluster_average_distance = 0
        for vector in sentence_vectors:
            current_cluster_average_distance += cosine_distance(cluster_centers[cluster_label], vector)
        average_cos_distance.append((cluster_label, current_cluster_average_distance/len(sentences)))
    average_cos_distance = sorted(average_cos_distance, key=lambda x:x[1], reverse=True)
    for d in average_cos_distance:
        print(d)
    # week5 作业 end

    for label, sentences in grouped_data.items():
        print("cluster %s :" % label)
        for i in range(min(10, len(sentences))):
            print(sentences[i].replace(" ", ""))
        print("---------")

if __name__ == "__main__":
    main()
