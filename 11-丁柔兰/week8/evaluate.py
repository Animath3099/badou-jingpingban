# -*- coding: utf-8 -*-
# =================================记录并显示预测的总数、正确数、错误数和准确率==============================
# 导入PyTorch模块和自定义的数据加载模块load_data
import torch
from loader import load_data

"""
模型效果测试
"""


class Evaluator:
    # 初始化函数接收三个参数：配置字典config，模型实例model，和日志记录器logger
    def __init__(self, config, model, logger):
        # 将传入的参数赋值给实例变量
        self.config = config
        self.model = model
        self.logger = logger
        # 加载验证数据集，不打乱顺序（因为在评估时通常不需要打乱数据）
        self.valid_data = load_data(config["valid_data_path"], config, shuffle=False)
        # 由于效果测试需要训练集当做知识库，再次加载训练集。
        # 事实上可以通过传参把前面加载的训练集传进来更合理，但是为了主流程代码改动量小，在这里重新加载一遍
        # 加载训练数据集，这里假设训练集数据会用作知识库。
        self.train_data = load_data(config["train_data_path"], config)
        # 初始化一个字典来记录测试的正确和错误统计
        self.stats_dict = {"correct": 0, "wrong": 0}  # 用于存储测试结果

    # 将知识库中的问题向量化，为匹配做准备
    # 每轮训练的模型参数不一样，生成的向量也不一样，所以需要每轮测试都重新进行向量化
    # 定义一个函数，用于将知识库中的问题转换为向量表示
    def knwb_to_vector(self):
        # 初始化两个字典，一个用于映射问题索引到标准问题索引，另一个用于存储问题的ID
        self.question_index_to_standard_question_index = {}
        self.question_ids = []
        # 遍历训练数据集中的知识库
        for standard_question_index, question_ids in self.train_data.dataset.knwb.items():
            # 将问题ID和其对应的标准问题索引添加到相应的字典中
            for question_id in question_ids:
                # 记录问题编号到标准问题标号的映射，用来确认答案是否正确
                self.question_index_to_standard_question_index[len(self.question_ids)] = standard_question_index
                self.question_ids.append(question_id)
        with torch.no_grad():  # 暂时禁用梯度计算，因为在评估阶段不需要更新模型权重
            # 将问题ID堆叠成一个张量
            question_matrixs = torch.stack(self.question_ids, dim=0)
            # 如果GPU可用，将数据移动到GPU上
            if torch.cuda.is_available():
                question_matrixs = question_matrixs.cuda()
            # 使用模型对问题进行编码，得到向量表示
            self.knwb_vectors = self.model(question_matrixs)
            # 将所有向量都作归一化 v / |v|
            # 归一化知识库中的向量表示
            self.knwb_vectors = torch.nn.functional.normalize(self.knwb_vectors, dim=-1)
        return

    # 定义一个eval函数，用于评估模型
    def eval(self, epoch):
        # 记录开始测试模型效果的日志
        self.logger.info("开始测试第%d轮模型效果：" % epoch)
        # 重置统计字典，以便新的评估周期
        self.stats_dict = {"correct": 0, "wrong": 0}  # 清空前一轮的测试结果
        # 将模型设置为评估模式，这通常会禁用dropout等训练时特有的行为
        self.model.eval()
        # 将知识库中的问题转换为向量表示
        self.knwb_to_vector()
        for index, batch_data in enumerate(self.valid_data):  # 遍历验证数据集
            if torch.cuda.is_available():  # 如果GPU可用，将批处理数据移动到GPU
                batch_data = [d.cuda() for d in batch_data]
            # 解包批处理数据中的输入ID和标签
            input_id, labels = batch_data  # 输入变化时这里需要修改，比如多输入，多输出的情况
            with torch.no_grad():  # 使用模型对验证集中的问题进行编码，得到向量表示
                test_question_vectors = self.model(input_id)  # 不输入labels，使用模型当前参数进行预测
            self.write_stats(test_question_vectors, labels)  # 调用write_stats函数来评价当前批次的结果
        self.show_stats()  # 显示整体评估结果
        return

    # 定义write_stats函数，用于计算并记录评估结果
    def write_stats(self, test_question_vectors, labels):
        # 断言以确保标签数量和测试问题向量数量相同
        assert len(labels) == len(test_question_vectors)
        # 遍历每个测试问题向量和其对应的标签
        for test_question_vector, label in zip(test_question_vectors, labels):
            # 通过一次矩阵乘法，计算输入问题和知识库中所有问题的相似度
            # test_question_vector shape [vec_size]   knwb_vectors shape = [n, vec_size]
            # 计算测试问题向量与知识库中所有问题向量的相似度
            res = torch.mm(test_question_vector.unsqueeze(0), self.knwb_vectors.T)
            # 找到最相似问题的索引
            hit_index = int(torch.argmax(res.squeeze()))  # 命中问题标号
            # 将问题索引转换成标准问题的索引
            hit_index = self.question_index_to_standard_question_index[hit_index]  # 转化成标准问编号
            # 根据预测是否正确，更新统计字典
            if int(hit_index) == int(label):
                self.stats_dict["correct"] += 1
            else:
                self.stats_dict["wrong"] += 1
        return

    # 定义show_stats函数，用于显示评估统计信息
    def show_stats(self):
        # 获取正确和错误的统计数值
        correct = self.stats_dict["correct"]
        wrong = self.stats_dict["wrong"]
        # 记录并显示预测的总数、正确数、错误数和准确率
        self.logger.info("预测集合条目总量：%d" % (correct + wrong))
        self.logger.info("预测正确条目：%d，预测错误条目：%d" % (correct, wrong))
        self.logger.info("预测准确率：%f" % (correct / (correct + wrong)))
        self.logger.info("--------------------")
        return
