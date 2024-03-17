# 基于训练好的词向量模型进行聚类
# 聚类采用Kmeans算法

# 增加计算类间平均距离并排序
import math
import re
import json
import jieba
import numpy as np
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from collections import defaultdict
from scipy.spatial.distance import pdist


# 输入模型文件路径
# 加载训练好的模型
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


# 将文本向量化
def sentences_to_vectors(sentences, model):
    vectors = []
    for sentence in sentences:
        words = sentence.split()  # sentence是分好词的，空格分开
        vector = np.zeros(model.vector_size)
        # 所有词的向量相加求平均，作为句子向量
        for word in words:
            try:
                vector += model.wv[word]
            except KeyError:
                # 部分词在训练中未出现，用全0向量代替
                vector += np.zeros(model.vector_size)
        vectors.append(vector / len(words))
    return np.array(vectors)


def main():
    model = load_word2vec_model("model.w2v")  # 加载词向量模型
    sentences = load_sentence("titles.txt")  # 加载所有标题
    vectors = sentences_to_vectors(sentences, model)  # 将所有标题向量化

    # n_clusters = int(math.sqrt(len(sentences))) 
    n_clusters = 7 # 指定聚类数量 5 7 10
    print("指定聚类数量：", n_clusters)
    kmeans = KMeans(n_clusters)  # 定义一个kmeans计算类
    kmeans.fit(vectors)  # 进行聚类计算

    sentence_label_dict = defaultdict(list)
    for sentence, label in zip(sentences, kmeans.labels_):  # 取出句子和标签
        sentence_label_dict[label].append(sentence)  # 同标签的放到一起

    # 计算类间距离
    sentence_label_mean_distance_list = []
    for label, sentences in sentence_label_dict.items():
        vectors_sentences = sentences_to_vectors(sentences, model)  # 将类内标题向量化
        row_distances = pdist(vectors_sentences)  # 计算类内距离和
        mean_distance = float(np.mean(row_distances))  # 求平均
        sentence_label_mean_distance_list.append([label, sentences, mean_distance])
    # 从小到大排序，距离越小越相似
    sentence_label_mean_distance_list_sort = sorted(sentence_label_mean_distance_list, key=lambda k: k[-1])
    print("排序后结果：")
    for i in sentence_label_mean_distance_list_sort:
        print("cluster %s :" % i[0])
        print("mean distance: %f" % i[2])
        for ii in range(min(10, len(i[1]))): 
            print(i[1][ii].replace(" ", ""))
        print("---------")
if __name__ == "__main__":
    main()
