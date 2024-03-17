import jieba
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from gensim.models import Word2Vec
import string
import json


def read_questions_from_json(json_path):
    """
    从JSON文件中读取问题，并返回问题的列表
    """
    with open(json_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    questions = [item["question"] for item in data]
    return questions


# 使用函数
json_path = "Python\\AI\\badou\\作业\\第五周\\questions.json"
questions = read_questions_from_json(json_path)
# 模拟中文文本数据
text_data = read_questions_from_json(json_path)


# 使用 jieba 进行中文分词
def preprocess_chinese(text):
    return list(jieba.cut(text))


processed_data = [preprocess_chinese(sentence) for sentence in text_data]

# 训练 Word2Vec 模型
model = Word2Vec(
    sentences=processed_data, vector_size=100, window=5, min_count=1, workers=2
)


# 向量化文本
def vectorize_sentence(sentence, model):
    word_vectors = [model.wv[word] for word in sentence if word in model.wv]
    return (
        np.mean(word_vectors, axis=0) if word_vectors else np.zeros(model.vector_size)
    )


sentence_vectors = np.array(
    [vectorize_sentence(sentence, model) for sentence in processed_data]
)

# 使用 KMeans 进行聚类
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(sentence_vectors)

# 提取聚类中心和类标签
cluster_centers = kmeans.cluster_centers_
labels = kmeans.labels_

# 创建 DataFrame 来存储数据点和它们的类别
df = pd.DataFrame(sentence_vectors)
df["Cluster"] = labels

# 计算每个点到其聚类中心的距离并排序
df["Distance"] = df.apply(
    lambda row: np.linalg.norm(
        row.drop("Cluster") - cluster_centers[int(row["Cluster"])]
    ),
    axis=1,
)
sorted_clusters = {
    cluster: df[df["Cluster"] == cluster].sort_values(by="Distance")
    for cluster in range(kmeans.n_clusters)
}

# 示例：显示第一个聚类的前几个排序点
print(sorted_clusters[0].head())
