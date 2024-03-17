#词向量文件

import os  #用于读取系统文件
from gensim.models import Word2Vec #从 Gensim 库中导入 Word2Vec 用于词嵌入的训练。
from nltk.tokenize import word_tokenize #用于将句子分割成单词或标记。
import nltk #导入 NLTK（Natural Language Toolkit）
nltk.download('punkt')
import jieba
import re #正则表达式

#我希望训练一个文件夹下的多个txt文本集合，所以先读取文件夹下的所有文本文件


def read_text(folder_path):
    texts = []

    for filename in os.listdir(folder_path):
        if filename.endswith('txt'):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                texts.append(file.read())
    return texts

def clean_text(text):
    #移除特殊字符和数字
    text = re.sub(r'\W+','',text)
    #移除纯数字
    text = re.sub(r'\d+','',text)
    #移除额外空白符
    text = re.sub(r's+','',text).strip()

    return text

def tokenize_text(text):
    return list(jieba.cut(text))

def train_modle(texts):

    processed_texts = texts

    vector_size = 100 #向量维度
    window = 5 #窗口大小
    min_count = 2 #忽略次数小于2次的次

    model = Word2Vec(sentences=processed_texts, vector_size=vector_size, window = window, min_count=min_count,sg=1,hs = 1,negative=5) #sg=1 指定skip-gram 不写默认CBOW

    model.save("word2vec.model")

    return model

def main():

  folder_path = 'E:\八斗学院录播\八斗课程-精品班\第五周\作业\财经'
  texts = read_text(folder_path)
  processed_texts = [tokenize_text(clean_text(text)) for text in texts]
  print('清洗完成')
  model = train_modle(processed_texts)
  print('建模完成')

  word = "技术"

  similar_word = model.wv.most_similar(word, topn=5)
  print(similar_word)

if __name__ == '__main__':
    main()