import math
import re
import json
import jieba
import numpy as np
from gensim.models import Word2Vec

from gensim.models.word2vec import LineSentence
from sklearn.cluster import KMeans
from collections import defaultdict

def Process_txt(path,write_path):
    '''
    输入文本文件的地址，对文本文件的每一行进行分词处理并写入另一个文件中
    :param path:
    :return:
    '''
    with open(path,'r',encoding='utf-8') as f:
        for line   in f:
            sentece=line.strip()
            sentence=jieba.cut(sentece)
            sentence="  ".join(sentence)
            with open(write_path,'a',encoding='utf-8') as f1:
                f1.writelines(sentence+"\n")


def load_model(path):
    model=Word2Vec.load(path)
    return model

def load_sentence(path):
    sentences=set()
    with open(path,'r',encoding='utf-8') as f:
        for line in f:
            sentence=line.strip()
            sentences.add(" ".join(jieba.cut(sentence)))

    return sentences

#将输入的句子向量化
def sentence_vector(model,sentences):
    vectors=[]
    for sentence in sentences:
        words=sentence.split()#用空格分开
        vector=np.zeros(model.vector_size)
        for word in words:
            try:
                vector+=model.wv[word]
            except KeyError:

                #未出现的词用全0替代
                vector+=np.zeros(model.vector_size)

        vectors.append(vector/len(words))
    return np.array(vectors)

def main():
    '''
    加载词模型和句子，向量化句子以后，使用K-means进行聚类。完成聚类以后，将同一类的句子放在一起。
    :return:
    '''
    model=load_model(r'E:\python\八斗\test.model')
    sentences=load_sentence(r"E:\badouFile\文件\week5 词向量及文本向量\week5 词向量及文本向量\titles.txt")

    vectors=sentence_vector(model,sentences)

    #n个句子分根号n类，是一个经验预设
    n_cluster=int(math.sqrt(len(sentences)))

    kmeans=KMeans(n_clusters=n_cluster)
    kmeans.fit(vectors)

    #
    #print(kmeans.labels_)
    # k-means.labels输出的是每个句子的标签，标签的取值范围是0到k-1

    sentence_label_dict=defaultdict(list)
    for sentence,label in zip(sentences,kmeans.labels_):
        sentence_label_dict[label].append(sentence)

    #计算类内距离
    density_dict=defaultdict(list)
    #这里就是迭代每个类标签和类标签对应的下标
    for vector_index,label in enumerate(kmeans.labels_):

        #取该类的类别中心
        center=kmeans.cluster_centers_[label]

        distance=consin_distance(vectors[vector_index],center)

        density_dict[label].append(distance)
    for label,distance_list in density_dict.items():

        #余弦距离取平均值
        density_dict[label]=np.mean(distance_list)

    distance_order=sorted(density_dict.items(),key=lambda x:x[1],reverse=True)

    for label,distance_avg in distance_order:
        print("cluster:{},avg_distance:{}".format(label,distance_avg))


def consin_distance(vec1,vec2):
    '''
    计算向量之间的余弦距离

    公式为 xi*yi/|x||y|
    :return:
    '''
    vec1=vec1/np.sqrt(np.sum(vec1**2))
    vec2=vec2/np.sqrt(np.sum(vec2**2))

    return np.sum(vec1*vec2)



if __name__ == '__main__':
    path_w=r"E:\badouFile\文件\week5 词向量及文本向量\week5 词向量及文本向量\text.txt"
    # Process_txt(r"E:\badouFile\文件\week5 词向量及文本向量\week5 词向量及文本向量\corpus.txt",path_w)
    # model = Word2Vec(
    #     LineSentence(open(path_w, 'r', encoding='utf8')),
    #     sg=0,
    #     window=3,
    #     min_count=1,
    #     workers=8
    # )
    # # 词向量保存
    # model.wv.save_word2vec_format('data.vector', binary=False)
    # # 模型保存
    # model.save('test.model')
    main()