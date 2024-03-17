import time
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from utils import *
from tqdm import tqdm

if torch.cuda.is_available():
    logger.info('use cuda')
else:
    logger.info('use cpu')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Config:
    def __init__(self):
        self.max_sequence_length = 256
        self.char_dim = 200
        self.hidden_size = 400
        self.num_gru_layers = 3
        self.dropout = 0.3
        self.train_epochs = 15
        self.batch_size = 125
        self.learning_rate = 5e-5
        self.log_step = 100
        self.train_data_path = 'dataset/train.txt'
        self.test_data_path = 'dataset/test.txt'
        self.model_save_path = 'model.pth'
        self.vocab_path = 'vocab.txt'


class WordSegmentationModel(nn.Module):
    def __init__(self, vocab, config):
        super(WordSegmentationModel, self).__init__()
        self.num_classes = 4  # BIES标签
        self.embedding = nn.Embedding(len(vocab), config.char_dim, padding_idx=0)  # 词嵌入
        # 双向GRU
        self.biGru = nn.GRU(input_size=config.char_dim,
                            hidden_size=config.hidden_size,
                            num_layers=config.num_gru_layers,
                            bidirectional=True,
                            batch_first=True,
                            dropout=config.dropout)
        self.classify = nn.Linear(config.hidden_size * 2, self.num_classes)  # 双向所以双倍hidden_size
        self.loss = nn.CrossEntropyLoss(ignore_index=-100).to(device)  # padding不计入损失

    def forward(self, x, y=None):
        # input_shape: (batch_size, sentence_length)
        x = self.embedding(x)  # output_shape: (batch_size, sentence_length, input_dim)
        outputs, _ = self.biGru(x)  # output_shape: (batch_size, sentence_length, hidden_size * 2)  双向所以双倍hidden_size
        y_pred = self.classify(outputs)
        if y is not None:
            return self.loss(y_pred.view(-1, self.num_classes), y.view(-1))
        else:
            return y_pred

class CWSDataset(Dataset):
    def __init__(self, data_path, vocab, config):
        super(CWSDataset, self).__init__()
        self.vocab = vocab
        self.data_path = data_path
        self.max_sequence_length = config.max_sequence_length
        self.load()

    def load(self):
        # 加载数据集
        self.data = []
        with open(self.data_path, 'r', encoding='utf8') as file:
            words = []
            labels = []
            for line in file:
                line = line.strip()
                if not line:
                    sequence = sentence_to_sequence(words, self.vocab)
                    label = labels_mapping(labels)
                    sequence, label = self.padding(sequence, label)
                    sequence = torch.LongTensor(sequence)
                    label = torch.LongTensor(label)
                    self.data.append([sequence, label])
                    words = []
                    labels = []
                    continue
                line_words = line.split('	')
                words.append(line_words[0])
                labels.append(line_words[1])

                # 部分数据，减少训练时间
                # if len(self.data) > 10000:
                #     break

    def padding(self, sequence, label):
        """
        用于词表转换后, 截断或者填充句子
        :return:
        """
        if len(sequence) >= self.max_sequence_length:
            return (sequence[:self.max_sequence_length],
                    label[:self.max_sequence_length])
        else:
            return (sequence + [0] * (self.max_sequence_length - len(sequence)),
                    label + [-100] * (self.max_sequence_length - len(label)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def sentence_to_sequence(sentence, vocab):
    """
    词表编码encode
    :param sentence:
    :param vocab:
    :return:
    """
    sequence = [vocab.get(char, vocab['[UNK]']) for char in sentence]
    return sequence

def sequence_to_text(sequence, vocab):
    """
    词表编码decode
    :param sequence:
    :param vocab:
    :return:
    """
    keys = list(vocab.keys())
    text = [keys[index] for index in sequence]
    return text

def padding_sequence(sequence, max_sequence_length):
    """
    序列填充
    :param sequence:
    :param max_sequence_length:
    :return:
    """
    if len(sequence) >= max_sequence_length:
        return sequence[:max_sequence_length]
    else:
        return sequence + [0] * (max_sequence_length - len(sequence))
def labels_mapping(labels):
    """
    序列标注标签体系(B、I、E、S),四个标签分别表示单字处理单词的起始、中间、终止位置或者该单字独立成词
    - B-CWS 对应 0
    - I-CWS 对应 1
    - E-CWS 对应 2
    - S-CWS 对应 3
    :param labels:
    :return:
    """
    mapping = []
    for label in labels:
        if label == 'B-CWS':
            mapping.append(0)
        elif label == 'I-CWS':
            mapping.append(1)
        elif label == 'E-CWS':
            mapping.append(2)
        elif label == 'S-CWS':
            mapping.append(3)
    return mapping

def build_vocab(vocab_path):
    """
    读取词表
    :param vocab_path:
    :return:
    """
    vocab = {}
    with open(vocab_path, 'r', encoding='utf8') as f:
        for i, line in enumerate(f):
            char = line.strip()
            vocab[char] = i
    return vocab

def build_dataset(data_path, vocab, config):
    """
    建立数据集
    :param data_path:
    :param vocab:
    :param config:
    :return:
    """
    dataset = CWSDataset(data_path, vocab, config)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, pin_memory=True)
    return dataloader

def get_seg_list(sentence, label):
    """
    输入字序列和标签序列，返回分词列表
    :param sentence:
    :param label:
    :return:
    """
    tmp_string = ''
    seg_list = []
    for i, p in enumerate(label):
        tmp_string += sentence[i]
        if p == 2 or p == 3:
            seg_list.append(tmp_string)
            tmp_string = ''
    return seg_list

def evaluate(model, vocab, config):
    model.eval()  # 评价模式
    dataloader = build_dataset(config.test_data_path, vocab, config)  # 加载测试数据集
    correct_all = []
    for x, y in tqdm(dataloader, desc='评估分词正确率'):
        x = x.to(device)
        y = y.to(device)

        with torch.no_grad():
            y_pred = model(x)
            for y_p, y_t, x_t in zip(y_pred, y, x):
                y_p = torch.argmax(y_p, dim=-1)
                # print(y_p)

                y_t = y_t.tolist()  # 在这里进行tolist()可以使用gpu计算，速度快很多，不要放进别的函数里，那样会切换至cpu
                y_p = y_p.tolist()

                # decode -> 词序列
                sentence = sequence_to_text(x_t, vocab)

                # 根据预测序列分词
                seg_t = get_seg_list(sentence, y_t)
                seg_p = get_seg_list(sentence, y_p)

                set_p = set(seg_p)
                set_t = set(seg_t)
                common = set_p.intersection(set_t)  # 求交集
                correct_all.append(len(list(common)) / len(seg_t))  # 计算每个句子的分词正确率  正确数 / 总词数

    logger.info(f'平均正确率: {np.mean(correct_all)}')
    return np.mean(correct_all)

def predict(input_strings, config):
    vocab = build_vocab(config.vocab_path)  # 建立词表
    model = WordSegmentationModel(vocab, config).to(device)  # 建模
    model.load_state_dict(torch.load(config.model_save_path))  # 加载权重
    model.eval()  # 评价模式
    for input_string in input_strings:
        x = sentence_to_sequence(input_string, vocab)  # encode
        x = padding_sequence(x, config.max_sequence_length)  # padding

        with torch.no_grad():
            x = torch.LongTensor([x])
            x = x.to(device)

            y_pred = model.forward(x)[0]  # 预测
            result = torch.argmax(y_pred, dim=-1)
            result = result.tolist()[:len(input_string)]
            for i, p in enumerate(result):  # 分词
                if p == 2 or p == 3:
                    print(input_string[i], end=' ')
                else:
                    print(input_string[i], end='')
            print()



def main():
    logger.info(parameters_dict)

    vocab = build_vocab(config.vocab_path)  # 建立词表
    data_loader = build_dataset(config.train_data_path, vocab, config)  # 加载训练数据集
    model = WordSegmentationModel(vocab, config).to(device)  # 建模
    optim = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)  # 选择优化器

    print('=========开始训练=========')
    log = []
    steps = 0
    for epoch in range(config.train_epochs):
        model.train()
        watch_loss = []
        start = time.time()  # 记录时间

        for x, y in data_loader:
            x = x.to(device)
            y = y.to(device)

            loss = model.forward(x, y)  # 计算损失
            loss.backward()  # 反向传播
            optim.step()  # 权重更新
            optim.zero_grad()  # 梯度清零

            watch_loss.append(loss.item())
            steps += 1
            if steps % config.log_step == 0:  # 每log_step步记录
                logger.info("=========\n第%d步平均loss:%f 耗时:%.2fs" % (steps, np.mean(watch_loss), time.time() - start))
                start = time.time()

        mean_loss = np.mean(watch_loss)

        logger.info("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))

        acc = evaluate(model, vocab, config)  # 评价

        log.append([acc, mean_loss])
        if epoch != 0:
            log_display(log)  # 每轮更新一次曲线

    # log_display(log)

    torch.save(model.state_dict(), config.model_save_path)  # 保存模型
    return





if __name__ == '__main__':
    # 初始化配置
    config = Config()

    # 超参
    config.train_epochs = 15
    config.batch_size = 125  # 125
    config.learning_rate = 5e-5
    config.log_step = 100

    config.max_sequence_length = 256
    config.char_dim = 200
    config.hidden_size = 400
    config.num_gru_layers = 3
    config.dropout = 0.3


    parameters_dict = config.__dict__  # 打印超参
    print(parameters_dict)

    # main()  # 训练

    input_strings = ["同时国内有望出台新汽车刺激方案",
                     "沪胶后市有望延续强势，经过两个交易日的强势调整后，昨日上海天然橡胶期货价格再度大幅上扬",
                     "中国政府将继续奉行独立自主的外交政策，在和平共处五项原则的基础上努力发展同世界各国的友好关系"]
    predict(input_strings, config)  # 预测
