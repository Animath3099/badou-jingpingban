# -*- coding: utf-8 -*-
import math
import os
import random
import torch
import re
import numpy as np
from collections import defaultdict

from transformers import BertTokenizer

from loader import load_data

"""
模型效果测试
"""


def sampling_strategy(prob_distribution):
    if random.random() > 0.3:
        strategy = "greedy"
    else:
        strategy = "sampling"
    if strategy == "greedy":
        return int(torch.argmax(prob_distribution))
    elif strategy == "sampling":
        prob_distribution = prob_distribution.cpu().numpy()
        return np.random.choice(list(range(len(prob_distribution))), p=prob_distribution)


class Evaluator:
    def __init__(self, config, model, logger):
        self.config = config
        self.model = model
        self.logger = logger
        self.tokenizer = BertTokenizer.from_pretrained(config.pretrain_model_path)

    def generate_sentence(self, openings):
        self.model.eval()
        with torch.no_grad():
            pred_char = ""
            # 生成了换行符，或生成文本超过20字则终止迭代
            while pred_char != "\n" and len(openings) <= 30:
                openings += pred_char
                x = self.tokenizer.encode(openings, add_special_tokens=False)
                x = torch.LongTensor([x])
                if torch.cuda.is_available():
                    x = x.cuda()
                y = self.model(x)[0][-1]
                index = sampling_strategy(y)
                pred_char = "".join(self.tokenizer.decode(index))
        return openings

    def calc_perplexity(self, sentence):
        prob = 0
        model = self.model
        window_size = self.config.window_size
        model.eval()
        with torch.no_grad():
            for i in range(1, len(sentence)):
                start = max(0, i - window_size)
                window = sentence[start:i]
                x = self.tokenizer.encode(window, max_length=window_size, padding="max_length", truncation=True)
                x = torch.LongTensor([x])
                target = sentence[i]
                target_index = self.tokenizer.ids_to_tokens.get(target)
                if torch.cuda.is_available():
                    x = x.cuda()
                pred_prob_distribute = model(x)[0][-1]
                target_prob = pred_prob_distribute[target_index]
                prob += math.log(target_prob, 10)
        return 2 ** (prob * (-1 / len(sentence)))

    def eval(self, epoch, save_weight=True):

        print(self.generate_sentence("让他在半年之前，就不能做出"))
        print(self.generate_sentence("李慕站在山路上，深深的呼吸"))
        if not save_weight:
            return
        else:
            model_path = os.path.join(self.config.model_path, f"model_{epoch}")
            torch.save(self.model.state_dict(), model_path)
            return
