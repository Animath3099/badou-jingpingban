# -*- coding: utf-8 -*-
import torch
import jieba
from model import SiameseNetwork
from config import Config


class SiameseLoader():
    def __init__(self):
        self.model = SiameseNetwork(Config)
        self.model.load_state_dict(torch.load('model_output/epoch_10.pth'))

        self.vocab = load_vocab()

    def encode_sentence(self, text):
        input_id = []
        if Config["vocab_path"] == "words.txt":
            for word in jieba.cut(text):
                input_id.append(self.vocab.get(word, self.vocab["[UNK]"]))
        else:
            for char in text:
                input_id.append(self.vocab.get(char, self.vocab["[UNK]"]))
        input_id = self.padding(input_id)

        # print(input_id)
        # sentence_vector = self.model(torch.LongTensor(input_id))

        return input_id
    def get_vector(self, text):
        input_id = self.encode_sentence(text)
        return torch.nn.functional.normalize(self.model(torch.LongTensor(input_id)).unsqueeze(0),dim=-1)
    def get_vectors(self,texts):
        all_text = []
        for text in texts:
            all_text.append(self.encode_sentence(text))
        return torch.nn.functional.normalize(self.model(torch.LongTensor(all_text)),dim=-1)


    # 补齐或截断输入的序列，使其可以在一个batch内运算
    def padding(self, input_id):
        input_id = input_id[:Config["max_length"]]
        input_id += [0] * (Config["max_length"] - len(input_id))
        return input_id


# 加载字表或词表
def load_vocab():
    token_dict = {}
    with open(Config['vocab_path'], encoding="utf8") as f:
        for index, line in enumerate(f):
            token = line.strip()
            token_dict[token] = index + 1  # 0留给padding位置，所以从1开始
    return token_dict
