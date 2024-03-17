# -*- coding: utf-8 -*-

import torch  # 导入 PyTorch 库，用于建立和训练深度学习模型
import os  # 导入 os 库，用于操作文件系统，如创建目录
import random  # 导入 random 库，用于生成随机数
import numpy as np  # 导入 NumPy 库，用于科学计算
import logging  # 导入日志模块，用于记录训练过程中的信息
from config import Config  # 从 config 文件中导入 Config 对象，包含模型配置
from model import TorchModel, choose_optimizer  # 从 model 文件中导入模型类和优化器选择函数
from evaluate import Evaluator  # 从 evaluate 文件中导入评估类
from loader import load_data  # 从 loader 文件中导入数据加载函数

# 配置日志模块，设置日志的级别为 INFO，并定义日志输出格式
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)  # 创建一个 logger 对象，用于输出日志信息

"""
模型训练主程序：
脚本导入必要的库，定义了一个 main 函数来处理模型训练的各个步骤，
包括模型的初始化、数据的加载、模型的训练和评估以及模型状态的保存。
最后，如果脚本直接运行，它会调用 main 函数并传入在 config.py 文件中定义的配置参数
"""


def main(config):  # 定义 main 函数，接收一个配置对象 config 作为参数
    # 创建保存模型的目录，如果目录不存在，则创建它
    if not os.path.isdir(config["model_path"]):
        os.mkdir(config["model_path"])

    # 加载训练数据，调用 load_data 函数，并传入训练数据路径和配置对象
    train_data = load_data(config["train_data_path"], config)

    # 加载模型，创建 TorchModel 实例，并传入配置对象
    model = TorchModel(config)

    # 检查是否可以使用 GPU，如果可以，则将模型转移到 GPU
    cuda_flag = torch.cuda.is_available()
    if cuda_flag:
        logger.info("gpu可以使用，迁移模型至gpu")
        model = model.cuda()

    # 加载优化器，调用 choose_optimizer 函数，并传入配置对象和模型
    optimizer = choose_optimizer(config, model)

    # 加载效果测试类，创建 Evaluator 实例，并传入配置对象、模型和 logger
    evaluator = Evaluator(config, model, logger)

    # 训练模型，循环遍历每个训练周期
    for epoch in range(config["epoch"]):
        epoch += 1  # 训练周期从 1 开始
        model.train()  # 将模型设置为训练模式
        logger.info("epoch %d begin" % epoch)  # 输出日志，表示训练周期开始
        train_loss = []  # 初始化一个列表，用于保存每个批次的损失

        # 遍历训练数据的每个批次
        for index, batch_data in enumerate(train_data):
            optimizer.zero_grad()  # 清空优化器的梯度

            # 如果可以使用 GPU，则将批次数据转移到 GPU
            if cuda_flag:
                batch_data = tuple(d.cuda() for d in batch_data)

            # 解包批次数据，得到输入 ID、注意力掩码、令牌类型 ID 和标签
            input_ids, attention_mask, token_type_ids, labels = batch_data

            # 前向传播，计算损失
            loss = model(input_ids, attention_mask, token_type_ids, labels)
            loss.backward()  # 反向传播，计算梯度
            optimizer.step()  # 根据梯度更新模型的参数
            train_loss.append(loss.item())  # 将损失添加到列表中

            # 如果是训练过程中的一半，则输出当前批次的损失
            if index % int(len(train_data) / 2) == 0:
                logger.info("batch loss %f" % loss)

        # 输出该训练周期的平均损失
        logger.info("epoch average loss: %f" % np.mean(train_loss))

        # 使用评估器评估当前训练周期的模型性能
        evaluator.eval(epoch)

    # 保存模型的状态字典到文件
    model_path = os.path.join(config["model_path"], "epoch_%d.pth" % epoch)
    torch.save(model.state_dict(), model_path)

    # 返回模型和训练数据
    return model, train_data


# 如果这个脚本是作为主程序运行，则调用 main 函数，并传入配置对象
if __name__ == "__main__":
    model, train_data = main(Config)
