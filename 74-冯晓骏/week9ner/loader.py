# -*- coding: utf-8 -*-
import json
from collections import defaultdict

import jieba
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from transformers import BertTokenizer


class DataGenerator():
    def __init__(self, config, path):
        self.config = config
        self.path = path
        self.tokenizer = BertTokenizer.from_pretrained(self.config['bert_path'])

        self.load_vocab()
        self.load_schema()
        self.sentences = []
        self.load()

        config['vocab_size'] = len(self.vocab_dict)

        # print(self.schema_dict)

    def load(self):
        self.data = []
        with open(self.path, 'r', encoding='utf-8') as f:
            segments = f.read().split('\n\n')

            for segment in segments:
                sentence = []
                labels = []
                for word in segment.split('\n'):
                    if word.strip() == '':
                        continue
                    char, label = word.split(' ')
                    sentence.append(char)
                    labels.append(self.schema_dict[label])

                self.sentences.append(''.join(sentence))

                input_ids = self.encode_sentence_by_vocab(sentence)
                labels = self.padding(labels, -1)
                self.data.append([torch.LongTensor(input_ids), torch.LongTensor(labels)])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def encode_sentence_by_vocab(self, sentence, padding=True):
        if self.config['use_bert'] or self.config['vocab_path'] == 'chars.txt' :
            input_ids = [self.vocab_dict.get(char, self.vocab_dict['[UNK]']) for char in sentence]
        else:
            input_ids = [self.vocab_dict.get(word, self.vocab_dict['[UNK]']) for word in jieba.cut(sentence)]

        if padding:
            input_ids = self.padding(input_ids)

        return input_ids

    def padding(self, input_ids, pad_token=0):
        input_ids = input_ids[:self.config['max_length']]
        input_ids = input_ids + (self.config['max_length'] - len(input_ids)) * [pad_token]
        return input_ids

    def load_vocab(self):
        # 加载词汇表
        self.vocab_dict = {}
        if self.config['use_bert']:
            self.vocab_dict = self.tokenizer.get_vocab()
        else:
            path = self.config['vocab_path']
            with open(path, 'r', encoding='utf-8') as f:
                for index, line in enumerate(f):
                    self.vocab_dict[line.strip()] = index + 1  # 0留给padding

    def load_schema(self):
        # 加载schema
        path = self.config['schema_path']
        with open(path, 'r', encoding='utf-8') as f:
            self.schema_dict = json.load(f)




def load_data(data_path, config, shuffle=True):

    data_generator = DataGenerator(config, data_path)
    data_loader = DataLoader(data_generator, batch_size=config['batch_size'], shuffle=shuffle)
    return data_loader


if __name__ == '__main__':
    from config import Config

    dl = load_data('ner_data/train', Config)
    for index,batch_data in enumerate(dl):
        print(batch_data)
        break
    # for sen,label in data:
    #     print(sen.shape,label.shape)

    # str = ['在', '阳', '光', '灿', '烂', '的', '日', '子', ',', '这', '个', '城', '市', '的', '天', '空', '总', '是', '很', '蓝', ',', '蓝', '得', '柔', '和', '明', '净', ',', '纤', '尘', '不', '染', ',', '因', '为', '擦', '洗', '这', '一', '方', '蓝', '天', '的', '除', '了', '雨', '水', ',', '还', '有', '音', '乐', '。']
    # for i in jieba.cut(''.join(str)):
    #     print(i)

    # for sen in data.dataset.vocab_dict.items():
    #     print(sen)
