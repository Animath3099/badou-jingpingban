# -*- coding: utf-8 -*-
import torch  # torch是PyTorch深度学习框架
import re  # re是正则表达式模块
import numpy as np  # numpy是一个强大的数学库，主要用于数组计算
from collections import defaultdict  # defaultdict是一个字典类的子类，它提供了一个默认值，用于字典所需的键不存在时
from loader import load_data  # load_data是一个导入函数，用于加载验证数据集，它可能在另一个名为loader的模块中定义

"""
模型效果测试
"""


# 定义了一个名为Evaluator的类，用于执行模型效果评估相关操作
class Evaluator:
    # config是配置信息，model是要评估的模型，logger是用于记录日志的对象
    def __init__(self, config, model, logger):
        # 这些行将传入的参数赋值给类的实例变量，以便在类的其他方法中使用
        self.config = config
        self.model = model
        self.logger = logger
        # 调用load_data函数加载验证数据集，并将结果赋值给self.valid_data
        self.valid_data = load_data(config["valid_data_path"], config, shuffle=False)
        # 从验证数据集中提取模式（schema），这可能是一个标签到索引的映射
        self.schema = self.valid_data.dataset.schema
        # 创建一个从索引到标签的映射字典，这是self.schema的逆映射
        self.index_to_label = dict((y, x) for x, y in self.schema.items())

    # 定义了一个名为eval的方法，用于执行模型的评估。它接受一个参数epoch，表示当前的训练轮次
    def eval(self, epoch):
        # 使用logger记录信息，表示开始了第epoch轮的模型评估
        self.logger.info("开始测试第%d轮模型效果：" % epoch)
        # 创建一个统计字典self.stats_dict，每个键对应一个标签，每个值是一个defaultdict，用于统计每个标签的预测情况
        self.stats_dict = dict(zip(self.schema.keys(), [defaultdict(int) for i in range(len(self.schema))]))
        # 将模型设置为评估模式。这通常会关闭诸如dropout等训练特有的层的特定行为
        self.model.eval()
        # 遍历验证数据集的每个批次。enumerate提供了批次的索引和数据
        for index, batch_data in enumerate(self.valid_data):
            # 获取当前批次的原始句子。这是通过索引来切片sentences列表实现的，范围取决于批次大小
            sentences = self.valid_data.dataset.sentences[
                        index * self.config["batch_size"]: (index + 1) * self.config["batch_size"]]
            if torch.cuda.is_available():  # 如果CUDA可用，则将批次数据移动到GPU上
                batch_data = [d.cuda() for d in batch_data]
            # 从批次数据中解包输入和标签
            input_id, labels = batch_data  # 输入变化时这里需要修改，比如多输入，多输出的情况
            # 在不计算梯度的情况下，使用模型对输入进行预测
            with torch.no_grad():
                pred_results = self.model(input_id)  # 不输入labels，使用模型当前参数进行预测
            # 调用write_stats方法写入统计信息
            self.write_stats(labels, pred_results, sentences)
        # 调用show_stats方法显示统计结果
        self.show_stats()
        return

    # 定义了一个名为write_stats的方法，用于计算和记录预测的统计数据
    def write_stats(self, labels, pred_results, sentences):
        # 确保标签、预测结果和句子长度相同，否则打印长度并抛出断言错误
        assert len(labels) == len(pred_results) == len(sentences), print(len(labels), len(pred_results), len(sentences))
        # 如果模型没有使用CRF层，则使用torch.argmax获取预测结果中的最大值索引
        if not self.config["use_crf"]:
            pred_results = torch.argmax(pred_results, dim=-1)
        # 遍历每个句子的真实标签、预测标签和句子本身
        for true_label, pred_label, sentence in zip(labels, pred_results, sentences):
            # 如果没有使用CRF层，则将预测标签从GPU移动到CPU，并转换成列表形式
            if not self.config["use_crf"]:
                pred_label = pred_label.cpu().detach().tolist()[:len(sentence)]
            # 将真实标签从GPU移动到CPU，并转换成列表形式
            true_label = true_label.cpu().detach().tolist()[:len(sentence)]
            # 遍历预测标签和真实标签
            for pred, gold in zip(pred_label, true_label):
                # 获取真实标签对应的文本标签
                key = self.index_to_label[gold]
                # 更新统计字典：如果预测正确，则增加correct计数，无论正确与否都增加total计数
                self.stats_dict[key]["correct"] += 1 if pred == gold else 0
                self.stats_dict[key]["total"] += 1
        return

    # 定义了一个名为show_stats的方法，用于显示模型的评估统计信息
    def show_stats(self):
        # 创建一个空列表，用于存储所有标签的准确率
        total = []
        # 遍历所有的标签
        for key in self.schema:
            # 计算每个标签的准确率，避免除以零
            acc = self.stats_dict[key]["correct"] / (1e-5 + self.stats_dict[key]["total"])
            # 使用logger记录每个标签的准确率
            self.logger.info("符号%s预测准确率：%f" % (key, acc))
            # 将准确率添加到total列表中
            total.append(acc)
        # 计算并记录所有标签准确率的平均值
        self.logger.info("平均acc：%f" % np.mean(total))
        # 记录一条分隔线，表示统计信息的结束
        self.logger.info("--------------------")
        return
