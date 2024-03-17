import math
import jieba
from gensim.models import Word2Vec
import numpy as np
from sklearn.cluster import KMeans
from collections import defaultdict

def load_w2v_model(path):
    """
    加载词向量模型
    :param path:
    :return:
    """
    return Word2Vec.load(path)

def load_sentences(path):
    """
    加载句子，并进行分词
    :param path:
    :return:
    """
    sentences = set()
    with open(path, 'r', encoding='utf8') as f:
        for line in f:
            sentence = line.strip()
            sentences.add(" ".join(jieba.cut(sentence)))
    return sentences

def sentences2vectors(sentences, model):
    """
    句子向量化
    :param sentences: 句子
    :param model: 词向量模型
    :return: 向量
    """

    vectors = []
    for sentence in sentences:
        words = sentence.split(" ")
        vector = np.zeros(model.vector_size)
        for word in words:
            try:
                vector += model.wv[word]
            except KeyError:
                vector += np.zeros(model.vector_size)
        vectors.append(vector / len(words))
    return np.array(vectors)

def calculate_vectors_distance(v1, v2):
    """
    计算向量间的欧式距离
    :param v1:
    :param v2:
    :return:
    """
    return np.linalg.norm(v1 - v2)

def get_sentences_distances_dict(sentences, model, centers):
    """
    获取句子和距离的map
    :param sentences:
    :param model:
    :param centers:
    :return:
    """
    cluster_vectors = sentences2vectors(sentences, model)  # 将类内所有标题向量化
    sentences_distances = {}
    for sentence, vector, center in zip(sentences, cluster_vectors, centers):
        distance = calculate_vectors_distance(vector, center)  # 计算距离
        sentences_distances[sentence] = distance
    return sentences_distances

def main():
    model = load_w2v_model("model.w2v")  # 加载词向量模型
    sentences = load_sentences("titles.txt")  # 加载所有标题
    vectors = sentences2vectors(sentences, model)  # 将所有标题向量化

    n_clusters = int(math.sqrt(len(sentences)))  # 指定聚类数量
    kmeans = KMeans(n_clusters)  # 定义Kmeans计算类
    kmeans.fit(vectors)  # 进行聚类计算

    sentence_label_dict = defaultdict(list)
    for sentence, label in zip(sentences, kmeans.labels_):  # 取出句子和标签
        sentence_label_dict[label].append(sentence)  # 同标签的放到一起

    for label, sentences in sentence_label_dict.items():
        sentences_distances = get_sentences_distances_dict(sentences, model, kmeans.cluster_centers_[label])  # 获取句子和距离的map
        sorted_sentences_distances = dict(sorted(sentences_distances.items(), key=lambda x: x[1]))  # 按距离从小到大排序
        print(f"---------{label}---------")


if __name__ == '__main__':
    main()