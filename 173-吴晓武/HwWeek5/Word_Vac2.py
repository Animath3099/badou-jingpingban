import os
import jieba
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import nltk
import re


def load_text(file_path):
    texts = []
    for filename in os.listdir(file_path):
        if filename.endswith('txt'):
            file_path = os.path.join(file_path, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                texts.append(file.read())
    return texts

def clean_text(text):
    text = re.sub(r'\W+' , "", text)   # \W 匹配 非字母数字的符号
    text = re.sub(r'\d+' , "", text)  # \d 匹配 所有数字
    text = re.sub(r'\s+',"",text) #\s+ 匹配任何空白字符（包括空格、制表符、换行符等）

    return text
def tokenlize_text(text):

    return list(jieba.cut(text))

def train_model(texts):

    processed_texts = texts

    vector_size = 100
    window = 10
    min_count = 2

    model = Word2Vec(sentences=processed_texts,vector_size=vector_size,window= window,min_count = min_count ,sg =1, hs = 1, negative = 5)

    model.save("WordVac2.model")
    return model

def main():
    folder_path = 'E:\八斗学院录播\八斗课程-精品班\第五周\作业\财经'

    texts = load_text(folder_path)

    process_text = [tokenlize_text(clean_text(text)) for text in texts]
    model = train_model(process_text)

    word ="金融"
    Recent_text = model.wv.most_similar(word, topn=5)

    print(Recent_text)

if __name__ == '__main__':
    main()





