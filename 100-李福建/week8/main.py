# -*- coding: utf-8 -*-

import torch
import os
import random
import os
import numpy as np
import logging
from config import Config
from model import SiameseNetwork, choose_optimizer
from evaluate import Evaluator, all_knwb_to_vector
from loader import load_data, load_schema, load_vocab
import jieba

logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

"""
模型训练主程序
"""

def main(config):
    #创建保存模型的目录
    if not os.path.isdir(config["model_path"]):
        os.mkdir(config["model_path"])
    #加载训练数据
    train_data = load_data(config["train_data_path"], config)
    #加载模型
    model = SiameseNetwork(config)
    # 标识是否使用gpu
    cuda_flag = torch.cuda.is_available()
    if cuda_flag:
        logger.info("gpu可以使用，迁移模型至gpu")
        model = model.cuda()
    #加载优化器
    optimizer = choose_optimizer(config, model)
    #加载效果测试类
    evaluator = Evaluator(config, model, logger)
    #训练
    for epoch in range(config["epoch"]):
        epoch += 1
        model.train()
        logger.info("epoch %d begin" % epoch)
        train_loss = []
        for index, batch_data in enumerate(train_data):
            optimizer.zero_grad()
            if cuda_flag:
                batch_data = [d.cuda() for d in batch_data]
            input_id1, input_id2, labels = batch_data   #输入变化时这里需要修改，比如多输入，多输出的情况
            loss = model(input_id1, input_id2, labels)
            train_loss.append(loss.item())
            # if index % int(len(train_data) / 2) == 0:
            #     logger.info("batch loss %f" % loss)
            loss.backward()
            optimizer.step()
        avg_loss = np.mean(train_loss)
        logger.info("epoch average loss: %f" % avg_loss)
        evaluator.eval(epoch)
    model_path = os.path.join(config["model_path"], "epoch_%d.pth" % epoch)
    torch.save(model.state_dict(), model_path)
    return


def encode_sentence(text):
    input_id = []
    vocab = load_vocab(Config["vocab_path"])
    if Config["vocab_path"] == "words.txt":
        for word in jieba.cut(text):
            input_id.append(vocab.get(word, vocab["[UNK]"]))
    else:
        for char in text:
            input_id.append(vocab.get(char, vocab["[UNK]"]))
    input_id = padding(input_id)
    return input_id

#补齐或截断输入的序列，使其可以在一个batch内运算
def padding(input_id):
    input_id = input_id[:Config["max_length"]]
    input_id += [0] * (Config["max_length"] - len(input_id))
    return input_id

def predict(model_path):
    load_data(Config["train_data_path"], Config)
    model = SiameseNetwork(Config)
    #model.load_state_dict(torch.load(model_path))
    model.eval()  # 测试模式
    knwb_vectors, question_index_to_standard_question_index = all_knwb_to_vector(Config, model, logger)
    schema = load_schema(Config["schema_path"])
    schema = {val: key for key, val in schema.items()}
    with torch.no_grad():
        question = "我的亲情号码有几个"
        input_id = encode_sentence(question)
        input_id = torch.LongTensor(input_id)
        input_id = model(input_id.unsqueeze(0))
        res = torch.mm(input_id.unsqueeze(0), knwb_vectors.T)
        hit_index = int(torch.argmax(res.squeeze()))  # 命中问题标号
        hit_label = question_index_to_standard_question_index[hit_index]  # 转化成标准问编号
        print(schema.get(hit_label))

if __name__ == "__main__":
    #main(Config)
    model_path = "model_output/epoch_10.pth"
    predict(model_path)




