# -*- coding: utf-8 -*-

import torch  # PyTorch深度学习框架
import random  # 生成随机数的random模块
import os  # 操作系统接口模块os
import numpy as np  # 科学计算库numpy
import logging  # 日志记录模块logging
from config import Config  # Config是配置文件
# SiameseNetwork和choose_optimizer是自定义的模型和优化器选择函数
from model import SiameseNetwork, choose_optimizer
from evaluate import Evaluator  # Evaluator是自定义的评估器
from loader import load_data  # load_data是数据加载函数

# 设置日志记录的基本配置，如日志级别和格式。然后，获取当前文件的日志记录器
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

"""
模型训练主程序
"""


# 定义了一个名为main的函数，它接收一个配置字典config作为参数
def main(config):
    # 创建保存模型的目录
    # 检查配置中指定的模型保存路径是否存在，如果不存在，则创建这个目录
    if not os.path.isdir(config["model_path"]):
        os.mkdir(config["model_path"])
    # 加载训练数据:调用load_data函数来加载训练数据
    train_data = load_data(config["train_data_path"], config)
    # 加载模型:实例化一个孪生网络模型，传入配置信息
    model = SiameseNetwork(config)
    # 标识是否使用gpu:检查CUDA是否可用（即GPU是否可用）
    cuda_flag = torch.cuda.is_available()
    if cuda_flag:  # 如果GPU可用，将模型转移到GPU上
        logger.info("gpu可以使用，迁移模型至gpu")
        model = model.cuda()
    # 加载优化器:基于配置选择一个优化器
    optimizer = choose_optimizer(config, model)
    # 加载效果测试类:实例化一个评估器对象，用于评估模型
    evaluator = Evaluator(config, model, logger)
    # 训练：开始训练模型，遍历配置中设置的训练轮数（epoch）
    for epoch in range(config["epoch"]):
        epoch += 1  # 训练轮数（epoch）
        model.train()  # 设置模型为训练模式
        logger.info("epoch %d begin" % epoch)  # 记录当前轮数的开始
        train_loss = []  # 初始化一个列表来记录训练过程中的损失
        for index, batch_data in enumerate(train_data):  # 遍历训练数据
            optimizer.zero_grad()  # 在每次梯度更新前重置梯度信息
            if cuda_flag:  # 如果GPU可用
                batch_data = [d.cuda() for d in batch_data]  # 将批数据移动到GPU上
            # 从批数据中解包输入和标签
            input_id1, input_id2, labels = batch_data  # 输入变化时这里需要修改，比如多输入，多输出的情况
            loss = model(input_id1, input_id2, labels)  # 将输入数据和标签传入模型，计算损失
            train_loss.append(loss.item())  # 将损失值添加到损失列表中
            # if index % int(len(train_data) / 2) == 0:
            #     logger.info("batch loss %f" % loss)
            loss.backward()  # 执行反向传播，计算梯度
            optimizer.step()  # 根据计算的梯度更新模型的参数
        logger.info("epoch average loss: %f" % np.mean(train_loss))  # 计算并记录本轮的平均损失
        evaluator.eval(epoch)  # 使用评估器评估当前训练的模型
    # 构造模型保存路径并保存模型的状态字典
    model_path = os.path.join(config["model_path"], "epoch_%d.pth" % epoch)
    torch.save(model.state_dict(), model_path)
    return


# 如果直接运行这个Python文件，则调用main函数并传入配置信息Config
if __name__ == "__main__":
    main(Config)
