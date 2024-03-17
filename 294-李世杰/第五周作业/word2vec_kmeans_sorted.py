from sklearn.cluster import KMeans
from collections import defaultdict
from gensim.models import Word2Vec
import jieba
import numpy as np
import math

def sentences_to_vectors(sentences,model):
    vectors = []
    unknow_word = set()
    for sentence in sentences:
        sentence =  sentence.split()
        vector = np.zeros(model.vector_size)
        # 所有词的向量相加求平均，作为句子向量
        for word in sentence:
            try:
                vector += model.wv[word]
            except KeyError:
                unknow_word.add(word)
                vector += np.zeros(model.vector_size)
        vectors.append(vector/len(sentence))
    print(f"unknow word size {len(unknow_word)}")
    return vectors

def load_sentence(path):
    # 获取去重后的句子 以set([],[],[])格式返回
    sentences  = set()
    with open(path,encoding='utf') as f:
        for line in f:
            sentences.add(" ".join(jieba.lcut(line.strip())))
    print("获取句子个数",len(sentences))
    return sentences
            
# 加载训练好的model
def load_word2vec_model(path):
    model = Word2Vec.load(path)
    return model

# def diy_sorted()

def main():
    model = load_word2vec_model(r"F:\BaiduNetdiskDownload\八斗课程-精品班\第五周\week5 词向量及文本向量\week5 词向量及文本向量\model.w2v")
    print("model加载完成：",model)
    sentences = load_sentence(r"F:\BaiduNetdiskDownload\八斗课程-精品班\第五周\week5 词向量及文本向量\week5 词向量及文本向量\titles.txt")
    vectors = sentences_to_vectors(sentences,model)
    n_clusters = int(math.sqrt(len(sentences)))
    print(f"聚类数量:{n_clusters}")
    kmeans = KMeans(n_clusters)
    kmeans.fit(vectors)
    print(kmeans.cluster_centers_.shape)
    print(len(kmeans.labels_))
    print(len(np.array(vectors)))
    
    label_vector_sentence = defaultdict(list)
    for vector, label, sentence in zip(vectors,kmeans.labels_,sentences):
        label_vector_sentence[label].append([vector,sentence])
    for label,vector_sentence in label_vector_sentence.items():
        print("cluster %s 共有%s 句:" % (label,len(vector_sentence)))
        # a = vector.reshape(1,-1)
        # b = kmeans.cluster_centers_[label].reshape(-1,1)
        # print(a.shape)
        # print(b.shape)
        # print(np.dot(a,b))
        # 用余弦相似度计算，并从大到小排序
        new_vector_sentence = sorted(vector_sentence,key=lambda x:np.dot(x[0],kmeans.cluster_centers_[label])/(np.linalg.norm(x[0])*np.linalg.norm(kmeans.cluster_centers_[label])))
        # 欧式距离
        # new_vector_sentence = sorted(vector_sentence,key=lambda x:np.linalg.norm((x[0],kmeans.cluster_centers_[label])))
        
        # 显示前5个
        for i in range(5):
            try:
                print(new_vector_sentence[i][1])
            except IndexError:
                break
        
            
if __name__ == "__main__":
    main()