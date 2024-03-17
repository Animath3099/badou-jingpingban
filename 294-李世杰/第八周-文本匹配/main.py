# -*- coding: utf-8 -*-

import torch
import os
import random
import os
import numpy as np
import logging
from config import Config
from model import SiameseNetwork, choose_optimizer
from evaluate import Evaluator
from loader import load_data
import copy
import csv
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
            if config['loss_type'] == "cosine_loss":
                input_id1, input_id2, labels = batch_data   #输入变化时这里需要修改，比如多输入，多输出的情况
                loss = model(sentence1=input_id1, sentence2=input_id2, target=labels)
            elif config['loss_type'] == "triple_loss":
                input_id1, input_id2, input_id3 = batch_data
                loss = model(sentence1=input_id1, sentence2=input_id2, sentence3=input_id3)
            train_loss.append(loss.item())
                
            # if index % int(len(train_data) / 2) == 0:
            #     logger.info("batch loss %f" % loss)
            loss.backward()
            optimizer.step()
        logger.info("epoch average loss: %f" % np.mean(train_loss))
        acc = evaluator.eval(epoch)
    model_path = os.path.join(config["model_path"], "epoch_%d.pth" % epoch)
    torch.save(model.state_dict(), model_path)
    return  acc

def write(Config,acc):
    file_name = "bert_train_information.csv"
    data_content = Config
    del data_content["vocab_path"]
    del data_content["pretrain_model_path"]
    del data_content["seed"]
    del data_content["valid_data_path"]
    del data_content["train_data_path"]
    del data_content["model_path"]
    
    data_content["acc"] = acc
    
    
    with open(file_name, "a", newline="", encoding="utf-8") as csv_file:
        # 创建 CSV writer 对象
        writer = csv.DictWriter(csv_file, fieldnames=data_content.keys())
        
        # 如果文件为空，则写入标题行
        if csv_file.tell() == 0:
            writer.writeheader()
        
        # 写入新的数据行
        writer.writerow(data_content)


seed = Config["seed"] 
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)        
        
if __name__ == "__main__":
    # main(Config)

    # for model in ["cnn"]:
    #     Config["model_type"] = model
    #     print("最后一轮准确率：", main(Config), "当前配置：", Config["model_type"])

    #对比所有模型
    #中间日志可以关掉，避免输出过多信息
    # 超参数的网格搜索
    # for model in ["layer"]:
    #     Config["model_type"] = model
    #     for lr in [1e-3, 1e-4,1e-5]:
    #         Config["learning_rate"] = lr
    #         for hidden_size in [64,128,256,512]:
    #             Config["hidden_size"] = hidden_size
    #             for batch_size in [2,4,8,16,32,64,128]:
    #                 Config["batch_size"] = batch_size
    #                 acc = main(Config)
    #                 write_config = copy.deepcopy(Config)
    #                 write(write_config,acc)
    #                 print("最后一轮准确率：", "当前配置：", Config)
    for model in ["bert"]:
        Config["model_type"] = model
        Config["bert"] = True
        for optimizer in ["adam","adamw","sgd"]:
            Config["optimizer"] = optimizer
            for lr in [1e-4,1e-5]:
                Config["learning_rate"] = lr
                for batch_size in [16,32,64,128]:
                    Config["batch_size"] = batch_size
                    for layer_hidden_size in [128,256,512]:
                        Config["layer_hidden_size"] = layer_hidden_size
                        for pool in ["avg","max"]:
                            Config["pool"] = pool
                            acc = main(Config)
                            write_config = copy.deepcopy(Config)
                            write(write_config,acc)
                            print("最后一轮准确率：", "当前配置：", Config)