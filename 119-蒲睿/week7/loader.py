import torch
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader


class DataGenerator:
    def __init__(self, data_path, config):
        self.config = config
        self.data_path = data_path
        self.config["class_num"] = 2  # 好评 or 差评
        if self.config["model_type"] == "bert":
            self.tokenizer = BertTokenizer.from_pretrained(config["pretrain_model_path"])
        self.vocab = load_vocab(self.config["vocab_path"])
        self.config["vocab_size"] = len(self.vocab)
        self.load()
        
    def load(self):
        self.data = []
        with open(self.data_path, "r", encoding="utf-8") as f:
            for line in f:
                row = line.strip().split(',', 1)
                if self.config["model_type"] == "bert":
                    input_comment = self.tokenizer.encode(row[1], max_length=self.config["max_length"], pad_to_max_length=True)
                else:
                    input_comment = self.encode_sentence(row[1])
                input_comment = torch.LongTensor(input_comment)
                label = torch.LongTensor([int(row[0])])
                self.data.append([input_comment, label])
        return
    
    
    def padding(self, string):
        string = string[:self.config["max_length"]]
        string += [0]* (self.config["max_length"] - len(string))
        return string
    
    
    def encode_sentence(self, text):
        input_id = []
        for char in text:
            input_id.append(self.vocab.get(char, self.vocab["[UNK]"]))  # get(key, value) 如果key不存在就返回指定值内容即[UNK]的编码
        input_id = self.padding(input_id)
        return input_id        
            
def load_vocab(vocab_path):
    token_dict = {}
    with open(vocab_path, encoding="utf-8") as f:
        for index, line in enumerate(f):
            token = line.strip()
            token_dict[token] = index + 1  # 0给padding
    return token_dict

def load_data(data_path, config, shuffle=True):
    data_generator = DataGenerator(data_path, config)
    data_loader = DataLoader(data_generator.data, batch_size=config["batch_size"], shuffle=shuffle)
    return data_loader


                