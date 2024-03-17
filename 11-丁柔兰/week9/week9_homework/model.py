# -*- coding: utf-8 -*-

import torch  # 导入 PyTorch 库，用于构建和训练神经网络
import torch.nn as nn  # 导入 PyTorch 库中的神经网络模块
from torch.optim import Adam, SGD  # 导入 PyTorch 库中的优化器模块，包括Adam和SGD
from torchcrf import CRF  # 导入 CRF 模块，用于添加条件随机场层
"""
建立网络模型结构：
这段代码定义了一个 PyTorch 神经网络模型，它结合了 BERT 作为特征提取器，以及一个可选的 CRF 层用于序列标注任务。
该模型适用于诸如命名实体识别（NER）等任务。代码中还包含了一个函数 choose_optimizer，它根据配置文件中的设置选择合适的优化器。
如果脚本直接运行，它将从 config 模块导入配置信息，并用这些配置信息实例化模型
"""

from transformers import BertModel  # 导入 transformers 库中的 BertModel，用于加载预训练的BERT模型

# 定义一个神经网络模型类，继承自 nn.Module
class TorchModel(nn.Module):
    # 初始化方法，接收一个配置字典 config 作为参数
    def __init__(self, config):
        super(TorchModel, self).__init__()  # 调用父类初始化方法
        self.bert = BertModel.from_pretrained(config["bert_path"])  # 加载预训练的BERT模型
        # 定义一个线性分类器，输入特征维度为BERT模型的隐藏层大小，输出特征维度为类别数量
        self.classifier = nn.Linear(self.bert.config.hidden_size, config["class_num"])
        # 初始化 CRF 层，class_num 为标签数量，batch_first=True表示输入的第一个维度是批量大小
        self.crf_layer = CRF(config["class_num"], batch_first=True)
        self.use_crf = config["use_crf"]  # 读取配置文件中是否使用CRF层的设置

    # 定义模型的前向传播方法
    def forward(self, input_ids, attention_mask, token_type_ids, labels=None):
        # 将输入传递给BERT模型，获取模型输出
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        sequence_output = outputs.last_hidden_state  # 获取序列输出
        logits = self.classifier(sequence_output)  # 将序列输出传递给分类器

        # 如果提供了标签，则计算损失
        if labels is not None:
            if self.use_crf:
                # 创建一个掩码，忽略索引为-100的位置
                mask = labels.ne(-100)
                # 计算CRF层的损失，where方法确保CRF层不会处理忽略索引位置
                loss = -self.crf_layer(logits, labels.where(mask, torch.tensor(0).to(labels.device)), mask=mask,
                                       reduction="mean")
                return loss  # 返回损失
            else:
                # 如果没有使用CRF层，则直接计算损失
                return self.loss(logits.view(-1, logits.shape[-1]), labels.view(-1))
        else:
            # 如果没有提供标签，则返回预测结果
            if self.use_crf:
                return self.crf_layer.decode(logits)  # 使用CRF层解码获取预测结果
            else:
                return logits  # 直接返回分类器的输出

# 定义一个根据配置选择优化器的函数
def choose_optimizer(config, model):
    optimizer = config["optimizer"]  # 从配置中读取优化器类型
    learning_rate = config["learning_rate"]  # 从配置中读取学习率
    if optimizer == "adam":
        return Adam(model.parameters(), lr=learning_rate)  # 创建Adam优化器
    elif optimizer == "sgd":
        return SGD(model.parameters(), lr=learning_rate)  # 创建SGD优化器

# 如果脚本作为主程序运行，则执行以下代码
if __name__ == "__main__":
    from config import Config  # 从 config 模块导入配置字典
    model = TorchModel(Config)  # 创建模型实例
