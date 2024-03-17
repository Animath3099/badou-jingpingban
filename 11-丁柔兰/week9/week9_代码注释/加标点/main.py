# -*- coding: utf-8 -*-

import torch
import os
import random
import os
import numpy as np
import logging
from config import Config
from model import TorchModel, choose_optimizer
from evaluate import Evaluator
from loader import load_data
#设置日志记录的基础配置，包括日志级别（INFO）和日志格式。创建一个日志记录器实例
logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

"""
模型训练主程序
"""
#定义了一个名为main的函数，它接受一个配置对象config作为参数
def main(config):
    #创建保存模型的目录:检查配置中指定的模型保存路径是否存在，如果不存在则创建该目录
    if not os.path.isdir(config["model_path"]):
        os.mkdir(config["model_path"])
    #加载训练数据
    train_data = load_data(config["train_data_path"], config)
    #加载模型
    model = TorchModel(config)
    # 标识是否使用gpu
    cuda_flag = torch.cuda.is_available()
    if cuda_flag:
        logger.info("gpu可以使用，迁移模型至gpu")
        model = model.cuda()
    #加载优化器
    optimizer = choose_optimizer(config, model)
    #加载效果测试类:实例化一个Evaluator对象，用于在训练过程中评估模型的性能
    evaluator = Evaluator(config, model, logger)
    #训练:开始训练循环，迭代配置中指定的轮数（epoch）
    for epoch in range(config["epoch"]):
        #增加轮数计数，将模型设置为训练模式，并记录当前轮数的开始
        epoch += 1
        model.train()
        logger.info("epoch %d begin" % epoch)
        train_loss = []#初始化一个列表用于存储每个批次的损失
        for index, batch_data in enumerate(train_data):#遍历训练数据集中的每个批次
            optimizer.zero_grad()#在每个批次开始前，清除之前的梯度信息
            if cuda_flag:#如果可以使用CUDA，将批次数据转移到GPU上
                batch_data = [d.cuda() for d in batch_data]
            #从批次数据中解包出输入数据和标签
            input_id, labels = batch_data   #输入变化时这里需要修改，比如多输入，多输出的情况
            loss = model(input_id, labels)#将输入数据和标签传递给模型，并计算损失
            loss.backward()#执行反向传播计算梯度
            optimizer.step()#执行优化步骤更新模型的权重
            train_loss.append(loss.item())#将当前步骤的损失添加到损失列表中
            if index % int(len(train_data) / 2) == 0:#在每个epoch的中间和末尾记录当前批次的损失
                logger.info("batch loss %f" % loss)
        #计算并记录整个轮数的平均损失
        logger.info("epoch average loss: %f" % np.mean(train_loss))
        evaluator.eval(epoch)#调用Evaluator对象的eval方法评估当前轮数的模型性能
    #构建模型保存路径，并保存模型的状态到文件
    model_path = os.path.join(config["model_path"], "epoch_%d.pth" % epoch)
    torch.save(model.state_dict(), model_path)
    return model, train_data#函数返回训练好的模型和训练数据

if __name__ == "__main__":
    model, train_data = main(Config)#启动模型的训练过程
