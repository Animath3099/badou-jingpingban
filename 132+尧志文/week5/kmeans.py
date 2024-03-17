import math
import re
import json
import jieba
import numpy as np
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from collections import defaultdict


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
    distince = [[x, 0] for x in range(n_clusters)]

    cluster_num = [0] * n_clusters
    for sen, vec, label in zip(sentences, vectors, kmeans.labels_):
        cluster_num[label] += 1
        distince[label][1] += np.linalg.norm(vec - kmeans.cluster_centers_[label])
    result = [[x[0], x[1]/y] for x, y in zip(distince, cluster_num)]
    result.sort(key=lambda x: x[1])
    print(result)
    sentence_label_dict = defaultdict(list)
    for sentence, label in zip(sentences, kmeans.labels_):  # 取出句子和标签
        sentence_label_dict[label].append(sentence)  # 同标签的放到一起
    for i in result:                 #按类内距离从小到大一次打印各个类
        print("cluster %s :" % i[0])
        for j in range(min(10, len(sentence_label_dict[i[0]]))):
            print(sentence_label_dict[i[0]][j].replace(" ", ""))
        print("---------")

    print("*******************************************************")
    large_distince = 0
    for i in result:      #将类内距离大于0.8的过滤
        if i[1] > 0.8:
            large_distince += 1
    final_cluster = n_clusters - large_distince
    kmeans = KMeans(final_cluster)  # 定义一个kmeans计算类
    kmeans.fit(vectors)
    sentence_label_dict = defaultdict(list)
    for sentence, label in zip(sentences, kmeans.labels_):  # 取出句子和标签
        sentence_label_dict[label].append(sentence)  # 同标签的放到一起
    for label, sentences1 in sentence_label_dict.items():
        print("cluster %s :" % label)
        for i in range(min(10, len(sentences1))):  # 随便打印几个，太多了看不过来
            print(sentences1[i].replace(" ", ""))
        print("---------")




if __name__ == "__main__":
    main()