import math

import numpy as np
from sklearn.cluster import KMeans
from gensim.models import Word2Vec
import jieba
from collections import defaultdict,OrderedDict
def corpus_to_sentences(path):
    '''
    语料分词
    :param path:语料路径
    :return: 分词列表
    '''
    sentences = []
    with open(path,encoding='utf-8') as f:
        for line in f:
            words = jieba.lcut(line)
            sentences.append(words)

    return sentences


def train_vector(corpus_path,vec_dim):
    '''
    训练词向量
    :param corpus_path: 训练词向量的语料路径
    :param vec_dim: 词向量维度
    :return: 词向量模型
    '''
    sentences = corpus_to_sentences(corpus_path)
    model = Word2Vec(sentences=sentences,vector_size=vec_dim,sg=1)
    model.save('model.w2v')

    print('词向量训练完成')
    return model

def load_vector_model(model_path):
    '''
    加载词向量模型
    :param model_path: 词向量路径
    :return: 词向量
    '''
    model = Word2Vec.load(model_path)
    print('词向量加载完成')
    return model

def sentences_to_vector(sentences,model):
    '''
    获取语料中所有句子的向量
    :param sentences: 所有句子
    :param model: 词向量模型
    :return: 所有句子向量，采用加和求平均方式
    '''
    vectors = []
    for sentence in sentences:
        vector = sentence_to_vector(sentence,model)
        vectors.append(vector)


    print(f'加载了{len(vectors)}个句子')
    return vectors
def sentence_to_vector(sentence,model):
    '''
    获得单个句子向量
    :param sentence: 句子
    :param model: 词向量模型
    :return: 句子向量，采用加和求平均方式
    '''
    vector = np.zeros(model.vector_size)
    for word in sentence.split():
        try:
            vector += model.wv[word]
        except KeyError:
            vector += np.zeros(model.vector_size)
    return vector/len(vector)

def load_sentence(path):
    '''
    对语料分词
    :param path: 语料路径
    :return: 分词后的数据
    '''
    sentences = []
    with open(path,encoding='utf-8') as f:
        for line in f:
            words = ' '.join(jieba.lcut(line.strip()))
            sentences.append(words)

    return sentences


def get_distance_to_center(setence,center):
    '''
    计算欧式距离
    :param setence: 句子向量
    :param center: 中心点向量
    :return: 距离
    '''
    sum = 0
    for x,y in zip(setence,center):
        sum += math.pow(x-y,2)
    return math.sqrt(sum)

if __name__ == '__main__':
    # vector_dim = 500
    # model = train_vector('corpus.txt',vector_dim)

    #截取的数量
    top = 20

    #加载词向量模型
    model = load_vector_model('model.w2v')

    #对语料预处理，获得分词的句子
    sentences = load_sentence('titles.txt')

    #获得句子的向量
    vectors = sentences_to_vector(sentences,model)

    #设置k值
    n_cluster = int(math.sqrt(len(vectors)))

    #聚类
    kmeans = KMeans(n_cluster)
    kmeans.fit(vectors)


    datas = defaultdict(dict)

    #遍历获取聚类每个点到中心点的距离
    for sentence, label in zip(sentences,kmeans.labels_):

        center = kmeans.cluster_centers_[label]

        #获取欧式距离
        distance = get_distance_to_center(sentence_to_vector(sentence,model),center)
        datas[label][sentence] = distance

    #对label进行排序
    datas = OrderedDict(sorted(datas.items(),key=lambda x:x[0]))
    for label,dict in datas.items():
        print('-------------------')
        print(label,':')

        # 根据距中心点的距离排序
        dict = OrderedDict(sorted(dict.items(),key=lambda x:x[1]))

        for sentence in list(dict.keys())[:top]:

            print(''.join(sentence.split()))
        print('-------------------')




