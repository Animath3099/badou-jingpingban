import os
import  jieba
from  gensim.models import Word2Vec
from nltk.tokenize import  word_tokenize
import nltk
import re

def load_file(file_path):
    texts = []
    for filename in os.listdir(file_path):
        if filename.endswith('txt'):
            file_path = os.path.join(file_path, filename)
            with open(file_path,'r',encoding='utf-8') as f:
                texts.append(f.read())
    return texts

def clean_text(text):
    text = re.sub(r'\W+','',text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', '', text)

    return text

def tokenlize_text(text):

    return list(jieba.cut(text))

def train_model(texts):
    processed_text = texts

    vector_size = 100
    window = 10
    min_count =3

    model = Word2Vec(sentences=processed_text,vector_size=vector_size,window = window,hs=1,negative=10)

    model.save("Word_Vec3.model")

    return model

def main():
    folder_path = 'E:\八斗学院录播\八斗课程-精品班\第五周\作业\财经'

    texts = load_file(folder_path)

    processed_text = [tokenlize_text(clean_text(text)) for text in texts]

    model = train_model(processed_text)

    word ="技术"

    similar_word = model.wv.most_similar(word, topn=10)

    print(similar_word)

if __name__ == '__main__':
    main()


