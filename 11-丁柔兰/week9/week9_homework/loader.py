# -*- coding: utf-8 -*-

import json  # 导入 json 库，用于读取和写入 JSON 文件。
import torch  # 导入 PyTorch 库，用于深度学习。
from torch.utils.data import Dataset, DataLoader  # 导入 PyTorch 数据集和数据加载器相关类。
from transformers import BertTokenizer  # 导入 transformers 库中的 BertTokenizer 类，用于 BERT 模型的文本分词。

"""
数据加载：
这段代码提供了一个用于加载和预处理用于命名实体识别（NER）任务的数据集的类 DataGenerator。
它继承自 PyTorch 的 Dataset 类，并使用 Hugging Face 的 transformers 库中的 BertTokenizer 对文本进行分词。
它还包含一个函数 load_data，用于创建 DataGenerator 的实例，并将其封装为一个 DataLoader 对象，用于迭代地加载数据。
代码的最后部分在作为主程序运行时将加载数据集
"""


# 定义一个数据生成器类，用来加载和预处理数据，它继承自 PyTorch 的 Dataset 类。
class DataGenerator(Dataset):
    # 初始化方法，接收数据文件路径和配置对象。
    def __init__(self, data_path, config):
        self.config = config  # 保存配置对象。
        self.path = data_path  # 保存数据文件路径。
        # 使用 transformers 中的 BertTokenizer 进行初始化，加载指定的 BERT 预训练模型分词器。
        self.tokenizer = BertTokenizer.from_pretrained(config["bert_path"])
        # 调用 load_schema 方法加载标签架构，这通常是一个 JSON 文件，映射标签到索引。
        self.schema = self.load_schema(config["schema_path"])
        self.data = []  # 初始化数据列表。
        self.load_data()  # 调用 load_data 方法加载数据。

    # 加载数据的方法。
    def load_data(self):
        # 打开数据文件，读取内容。
        with open(self.path, "r", encoding="utf8") as f:
            # 以空行为分隔，分割数据为多个段落。
            segments = f.read().strip().split("\n\n")
            self.sentences = []  # 初始化句子列表。
            # 遍历每个段落。
            for segment in segments:
                sentence, labels = [], []  # 初始化句子和标签列表。
                # 分割段落为多行，每行包含字符和标签。
                for line in segment.split("\n"):
                    if line:
                        char, label = line.split()  # 分割行为字符和标签。
                        sentence.append(char)  # 添加字符到句子列表。
                        labels.append(self.schema[label])  # 将标签映射到索引并添加到标签列表。
                # 将句子列表合并为字符串并添加到句子列表。
                self.sentences.append(''.join(sentence))
                # 使用 BERT 分词器对句子进行编码。
                encoded_inputs = self.tokenizer(sentence,
                                                is_split_into_words=True,
                                                max_length=self.config["max_length"],
                                                truncation=True,
                                                padding='max_length',
                                                return_tensors="pt")
                # 提取编码后的输入 ID、注意力掩码、令牌类型 ID。
                input_ids = encoded_inputs["input_ids"].squeeze()
                attention_mask = encoded_inputs["attention_mask"].squeeze()
                token_type_ids = encoded_inputs["token_type_ids"].squeeze()
                # 调用 encode_labels 方法将标签编码为模型可以理解的格式。
                label_ids = self.encode_labels(labels)
                # 将编码后的数据添加到数据列表。
                self.data.append((input_ids, attention_mask, token_type_ids, torch.LongTensor(label_ids)))

    # 编码标签的方法。
    def encode_labels(self, labels):
        # 保证第一个标签是有效标签，而不是忽略索引 -100。
        label_ids = [self.schema.get(labels[0], 0)]
        # 为剩余标签执行相同操作。
        for label in labels[1:]:
            label_ids.append(self.schema.get(label, -100))
        # 截断或填充标签列表，使其长度等于配置中的最大长度。
        label_ids = label_ids[:self.config["max_length"]]
        padding_length = max(self.config["max_length"] - len(label_ids), 0)
        label_ids.extend([-100] * padding_length)
        return label_ids

    # 获取数据集长度的方法。
    def __len__(self):
        return len(self.data)

    # 获取指定索引的数据的方法。
    def __getitem__(self, index):
        return self.data[index]

    # 加载标签架构的方法。
    def load_schema(self, path):
        with open(path, encoding="utf8") as f:
            return json.load(f)

# 加载字或词表的函数。
def load_vocab(vocab_path):
    token_dict = {}
    with open(vocab_path, encoding="utf8") as f:
        for index, line in enumerate(f):
            token = line.strip()
            token_dict[token] = index + 1  # 索引从 1 开始，0 留给 padding。
    return token_dict

# 使用 PyTorch 的 DataLoader 类封装数据的函数。
def load_data(data_path, config, shuffle=True):
    dg = DataGenerator(data_path, config)  # 创建 DataGenerator 实例。
    dl = DataLoader(dg, batch_size=config["batch_size"], shuffle=shuffle)  # 创建 DataLoader 实例。
    return dl

# 如果该脚本作为主程序运行，则创建 DataGenerator 实例并加载数据。
if __name__ == "__main__":
    from config import Config  # 从 config 模块导入配置信息。
    dg = DataGenerator("../ner_data/train.txt", Config)
