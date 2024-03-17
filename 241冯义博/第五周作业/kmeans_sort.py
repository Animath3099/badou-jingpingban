import collections

from gensim.models import Word2Vec
import jieba
import numpy as np
from sklearn.cluster import KMeans
import math



def load_word2vec_mode(path):
    mode = Word2Vec.load(path)
    return mode


def load_sentence(path):
    data = set()
    with open(path, encoding='utf-8') as f:
        for line in f:
            sentence = line.strip()
            data.add(" ".join(jieba.cut(sentence)))
        print("句子数量 %s" % len(data))
        return data


def handle_vector(mode, datas):
    vectors = []
    for sentence in datas:
        sentence_vec = np.zeros(mode.vector_size)
        words = sentence.split()
        for word in words:
            try:
                sentence_vec += mode.wv[word]
            except KeyError:
                sentence_vec += np.zeros(mode.vector_size)
        vectors.append(sentence_vec / len(words))
    return vectors



def sort(label_dict, vec_dict, center_dict):
    """
    按欧式距离排序取前10
    :param label_dict:
    :param vec_dict:
    :param center_dict:
    :return:
    """
    label_s_dict = dict()
    for label, sentences in label_dict.items():
        min_l = []
        for s in sentences:
            vec = vec_dict[s]
            center = center_dict[label]
            distance = get_distance(center, vec)
            s_d = tuple([s, distance])
            if len(min_l) < 10:
                min_l.append(s_d)
            else:
                max_s = min_l[9]
                if distance < max_s[1]:
                    min_l.remove(max_s)
                    min_l.append(s_d)
            min_l.sort(key=lambda k: k[1])
        label_s_dict[label] = min_l
    return label_s_dict





def get_distance(center, position):
    return np.sqrt(np.sum(np.square(position - center)))


def main():
    mode = load_word2vec_mode("model.w2v")
    datas = load_sentence('titles.txt')
    # 所有句子的向量
    vectors = handle_vector(mode, datas)
    n_clusters = int(math.sqrt(len(datas)))
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(vectors)
    sentence_vec = dict()
    sentence_label_dict = collections.defaultdict(list)
    label_center_dict = dict()
    for sentence, vec in zip(datas, vectors):
        sentence_vec[sentence] = vec
    for sentence, label in zip(datas, kmeans.labels_):
        sentence_label_dict[label].append(sentence)
    for label, label_vec in enumerate(kmeans.cluster_centers_):
        label_center_dict[label] = label_vec
    label_s_dict = sort(sentence_label_dict, sentence_vec, label_center_dict)
    for label, sentences in label_s_dict.items():
        print("cluster %s :" % label)
        for i in sentences:
            print(i[0].replace(" ", ""))
            print(i[1])
        print("---------")






if __name__ == "__main__":
    main()