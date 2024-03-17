"""
训练词向量模型并预测词的近义词
"""
import jieba
from gensim.models import Word2Vec
import numpy as np 
import math
from sklearn.cluster import KMeans
from collections import defaultdict


# 训练模型
def train_word_vec(corpus, dim):
    # 创建word2vec模型，sg=1是skip-gram(用中间词预测两边词)，其他情况是CBOW(用两边词预测中间词)
    model = Word2Vec(corpus, vector_size=dim, sg=1)
    model.save('word2vec.model')
    return model

sentences = []
with open('corpus.txt', encoding='utf8') as f:
    for line in f:
        sentences.append(jieba.lcut(line))
train_word_vec(sentences, 100)

"""
基于训练好的模型进行聚类分析
1 加载模型
2 加载文本
3 文本中的句子进行向量化
4 定义聚类的类，进行分析
5 结果输出
"""

# 加载训练好的模型
def load_model(path):
    model = Word2Vec.load(path)
    return model

# 加载文本:文本用jieba进行分词，每个词添加到sentences的集合里面
# 用集合可以保证不会有重复的句子，最终返回的sentences的集合中的每个元素是由词和空格组成的一个字符串
def load_corpus(path):
    sentences = set()
    with open(path, encoding='utf8') as f:
        for line in f:
            sentence = line.strip()
            sentences.add(' '.join(jieba.cut(sentence)))
        print(f'句子的数量是：{len(sentences)}')
    return sentences

# 将文本向量化
def sen_to_vec(sentences, model):
    vectors = []
    for sentence in sentences:
        words = sentence.split()
        # 用numpy创建一个值都是0的向量，向量的大小和训练好的模型的词向量大小一致
        # .vector_size表示每个词向量的维度，是Word2Vec模型训练完成后的一个属性
        # 每个词向量的维度在训练Word2Vec模型的时候定义好的，这里是100
        vector = np.zeros(model.vector_size)
        # 采用句子中的词的向量加和求平均，作为句子的向量
        for word in words:
            try:
                # 模型训练好之后，wv就是词向量
                # 获取特定词的词向量 vector = model.wv['computer']
                # 计算两个词之间的相似度 similarity = model.wv.similarity('woman', 'man')
                # 找到最相似的词，默认输出10个最相似的，参数是topn=，similar_words = model.wv.similarity('machine')
                vector += model.wv[word]
            except KeyError:
                # 防止有的词没有在词向量中出现过，用0去代替没有见过的词
                vector += np.zeros(model.vector_size)
        vectors.append(vector / len(words))
    # 把所有的句子向量转换成np数组的形式
    return np.array(vectors)

# 过程
# 加载模型
model = load_model("word2vec.model")
# 加载文本
sentences = load_corpus('titles.txt')
# 句子向量化
vectors = sen_to_vec(sentences, model)

# 指定聚类的数量，使用经验公式
n_clusters = int(math.sqrt(len(sentences)))
print(f'聚类的数量是：{n_clusters}')
# 定义聚类模型，入参是聚类的数量
kmeans = KMeans(n_clusters)
# 进行聚类分析，入参是所有的向量
kmeans.fit(vectors)

# print(kmeans.cluster_centers_) # 这个是每个聚类的中心点
# print('---------------')
# print(kmeans.labels_) # 这个是每句话对应的聚类标签,numpy.ndarray
# # 结果输出
# # 用defaultdict输出内容：key：聚类标签，list：这个标签中的句子(也就是文本中一个个的标题)
# sen_class_dict = defaultdict(list)
# # .labels_是KMeans执行过fit()方法之后具有的属性，就是标签
# for sentence, label in zip(sentences, kmeans.labels_):
#     sen_class_dict[label].append(sentence)

# 句子：sentences[i],句子对应的向量：vectors[i]
# i->labels->centers->dis
# labels[i]:第i个句子对应的label
# centers[labels[i]]:第i个句子对应的center
# vector[i]:第i个句子对应的vector
distances = []
for i in range(len(sentences)):
    label = kmeans.labels_[i]
    center = kmeans.cluster_centers_[label]
    vector = vectors[i]
    # 计算center和vector的距离
    dis = np.linalg.norm(center - vector)
    distances.append(dis)
sen_order_dict = defaultdict(list)
for sentence, distance, label in zip(sentences, distances, kmeans.labels_):
    sen_order_dict[label].append([sentence, distance])
for label in sen_order_dict:
    sen_order_dict[label] = sorted(sen_order_dict[label], reverse=True, key=lambda x:x[1])

# 打印结果
for label, sen_dic in sen_order_dict.items():
    print(f'类别标签是：{label}')
    for i in range(min(3, len(sen_dic))): # 最多打印10个, sen_dic是每个类别的句子数量
        print(sen_dic[i][0].replace(' ', ''))