# -*- coding: utf-8 -*-

import torch  # 导入PyTorch库，用于深度学习任务。
import re  # 导入正则表达式库，用于文本匹配。
import numpy as np  # 导入NumPy库，用于数学运算。
from collections import defaultdict  # 导入defaultdict，它是一个带有默认值的字典。
from loader import load_data  # 从loader模块导入load_data函数，用于加载数据。

"""
模型效果测试
"""

# 定义一个评估器类，用于评估模型的效果。
class Evaluator:
    # 初始化方法，接收配置对象、模型实例和日志记录器。
    def __init__(self, config, model, logger):
        self.config = config  # 保存配置信息。
        self.model = model  # 保存模型实例。
        self.logger = logger  # 保存日志记录器。
        # 加载验证集数据。
        self.valid_data = load_data(config["valid_data_path"], config, shuffle=False)

    # 评估模型的方法，接收一个训练周期数作为参数。
    def eval(self, epoch):
        self.logger.info("开始测试第%d轮模型效果：" % epoch)  # 记录日志。
        # 初始化统计字典，用于记录不同实体类型的识别情况。
        self.stats_dict = {"LOCATION": defaultdict(int),
                           "TIME": defaultdict(int),
                           "PERSON": defaultdict(int),
                           "ORGANIZATION": defaultdict(int)}
        self.model.eval()  # 将模型设置为评估模式。
        # 遍历验证集数据。
        for index, batch_data in enumerate(self.valid_data):
            # 从数据集中获取对应的句子。
            sentences = self.valid_data.dataset.sentences[index * self.config["batch_size"]: (index+1) * self.config["batch_size"]]
            # 如果GPU可用，则将数据移至GPU。
            if torch.cuda.is_available():
                batch_data = [d.cuda() for d in batch_data]
            # 解包批次数据。
            input_ids, attention_mask, token_type_ids, labels = batch_data
            # 不计算梯度，进行推理。
            with torch.no_grad():
                pred_results = self.model(input_ids, attention_mask, token_type_ids)
            # 写入统计信息。
            self.write_stats(labels, pred_results, sentences)
        # 显示统计信息。
        self.show_stats()
        return

    # 将模型预测和实际标签的统计信息写入字典的方法。
    def write_stats(self, labels, pred_results, sentences):
        # 确保标签、预测结果和句子长度一致。
        assert len(labels) == len(pred_results) == len(sentences)
        # 如果不使用CRF，则使用argmax获取最可能的标签。
        if not self.config["use_crf"]:
            pred_results = torch.argmax(pred_results, dim=-1)
        # 遍历每个句子及其真实和预测标签。
        for true_label, pred_label, sentence in zip(labels, pred_results, sentences):
            if not self.config["use_crf"]:
                pred_label = pred_label.cpu().detach().tolist()  # 将预测结果转换为列表。
            true_label = true_label.cpu().detach().tolist()  # 将真实标签转换为列表。
            # 解码标签为实体。
            true_entities = self.decode(sentence, true_label)
            pred_entities = self.decode(sentence, pred_label)
            # 更新统计字典。
            for key in ["PERSON", "LOCATION", "TIME", "ORGANIZATION"]:
                self.stats_dict[key]["正确识别"] += len([ent for ent in pred_entities[key] if ent in true_entities[key]])
                self.stats_dict[key]["样本实体数"] += len(true_entities[key])
                self.stats_dict[key]["识别出实体数"] += len(pred_entities[key])
        return

    # 显示模型预测的统计信息的方法。
    def show_stats(self):
        F1_scores = []
        # 计算每个实体类型的精确度、召回率和F1分数，并记录日志。
        for key in ["PERSON", "LOCATION", "TIME", "ORGANIZATION"]:
            precision = self.stats_dict[key]["正确识别"] / (1e-5 + self.stats_dict[key]["识别出实体数"])
            recall = self.stats_dict[key]["正确识别"] / (1e-5 + self.stats_dict[key]["样本实体数"])
            F1 = (2 * precision * recall) / (precision + recall + 1e-5)
            F1_scores.append(F1)
            self.logger.info("%s类实体，准确率：%f, 召回率: %f, F1: %f" % (key, precision, recall, F1))
        # 计算所有实体类型的宏平均F1分数，并记录日志。
        self.logger.info("Macro-F1: %f" % np.mean(F1_scores))
        # 计算所有实体类型的微平均F1分数，并记录日志。
        correct_pred = sum([self.stats_dict[key]["正确识别"] for key in ["PERSON", "LOCATION", "TIME", "ORGANIZATION"]])
        total_pred = sum([self.stats_dict[key]["识别出实体数"] for key in ["PERSON", "LOCATION", "TIME", "ORGANIZATION"]])
        true_enti = sum([self.stats_dict[key]["样本实体数"] for key in ["PERSON", "LOCATION", "TIME", "ORGANIZATION"]])
        micro_precision = correct_pred / (total_pred + 1e-5)
        micro_recall = correct_pred / (true_enti + 1e-5)
        micro_f1 = (2 * micro_precision * micro_recall) / (micro_precision + micro_recall + 1e-5)
        self.logger.info("Micro-F1 %f" % micro_f1)
        self.logger.info("--------------------")
        return

    # 根据一个给定的标签序列解码出实体的方法。
    def decode(self, sentence, labels):
        # 将标签序列转换为字符串。
        labels = "".join([str(x) for x in labels[:len(sentence)]])
        results = defaultdict(list)
        # 使用正则表达式查找不同类型实体在标签序列中的位置，并抽取对应的文本作为实体。
        for location in re.finditer("(04+)", labels):
            s, e = location.span()
            results["LOCATION"].append(sentence[s:e])
        for location in re.finditer("(15+)", labels):
            s, e = location.span()
            results["ORGANIZATION"].append(sentence[s:e])
        for location in re.finditer("(26+)", labels):
            s, e = location.span()
            results["PERSON"].append(sentence[s:e])
        for location in re.finditer("(37+)", labels):
            s, e = location.span()
            results["TIME"].append(sentence[s:e])
        return results  # 返回解码后的实体字典。
