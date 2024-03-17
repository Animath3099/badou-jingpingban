from model import TorchModel
from config import Config
import torch
import torch.nn as nn
from collections import defaultdict
import re

class Predict:
    def __init__(self, config, model_path):
        self.config = config
        self.vocab = self.load_vocab(config["vocab_path"])
        self.model = TorchModel(config)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

    # 加载字表或词表
    def load_vocab(self, vocab_path):
        token_dict = {}
        with open(vocab_path, encoding="utf8") as f:
            for index, line in enumerate(f):
                token = line.strip()
                token_dict[token] = index + 1  # 0留给padding位置，所以从1开始
        self.config["vocab_size"] = len(token_dict)
        return token_dict


    def predict(self, text):
        input_id = []
        for char in text:
            input_id.append(self.vocab.get(char, self.vocab["[UNK]"]))
        input_id = torch.LongTensor([input_id])
        with torch.no_grad():
            result = self.model(input_id)
        return self.decode(text, result)

    def decode(self, sentence, result):
        results = defaultdict(set)
        for labels in result:
            labels = "".join([str(x) for x in labels[:len(sentence)]])
            for location in re.finditer("(04+)", labels):
                s, e = location.span()
                results["LOCATION"].add(sentence[s:e])
            for location in re.finditer("(15+)", labels):
                s, e = location.span()
                results["ORGANIZATION"].add(sentence[s:e])
            for location in re.finditer("(26+)", labels):
                s, e = location.span()
                results["PERSON"].add(sentence[s:e])
            for location in re.finditer("(37+)", labels):
                s, e = location.span()
                results["TIME"].add(sentence[s:e])
        return results

def loadValidData(valid_path):
    sentence = ""
    with open(valid_path, encoding="utf8") as f:
        for line in f:
            sentence += line
    return sentence

if __name__ == "__main__":
    pr = Predict(Config, "model_output/lstm_epoch_20.pth")

    sentence = loadValidData("valid.txt")

    res_list = pr.predict(sentence)

    print("lstm 模型实体抽取结果如下：")
    for label, res in res_list.items():
        print("实体%s结果: %s" % (label, res))


