# coding:utf8
import json
import os
import random
import signal
import time

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader, random_split

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
from sklearn.metrics import f1_score, recall_score

device = "cuda:0" if torch.cuda.is_available() else "cpu"
np.set_printoptions(threshold=np.inf)


def set_seed(seed=None):
    # 根据种子设置随机数
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.cuda.manual_seed(seed)
            # 保证在输入和网络结构不变的情况下， 网络行为的不变性
            # torch.backends.cudnn.deterministic = True
            # torch.backends.cudnn.benchmark = False
    else:
        torch.manual_seed(torch.initial_seed())
        np.random.seed()
        random.seed()
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(torch.initial_seed())
            torch.cuda.manual_seed(torch.initial_seed())
            # torch.backends.cudnn.deterministic = True
            # torch.backends.cudnn.benchmark = False


class Random_Dataset(Dataset):
    def __init__(
        self,
        target_index,
        num_samples,
        sample_length,
        embedding_dims,
        file_name,
        seed=None,
    ):
        """
        定义一个产生随机训练数据的数据集
        input_para:
                target_index:网络中需要处理的字符对应的下标
                num_samples: 需要生成的数据集中数据量
                sample_length: 单个数据的长度
                embedding_dims: embedding层中单个字符对应的向量的维度
                file_name: 存储词表的文件名
        """
        super(Random_Dataset, self).__init__()
        self.num_samples = num_samples
        self.file_name = file_name
        self.seed = seed
        self.sample_length = sample_length
        self.target_index = target_index
        self.embedding_dims = embedding_dims
        (
            self.data,
            self.label,
            self.voca_list,
            self.max_len,
            self.voca_max_len,
        ) = self._build_samples()
        self.num_embeddings = self.voca_max_len
        self.vector, self.label = self._padding()

    def __len__(self):
        return len(self.vector)

    def __getitem__(self, index):
        # 这个要根据调用的损失函数进行调整
        data = torch.tensor(self.vector[index], dtype=torch.long)  # 整数类型
        label = torch.tensor(self.label[index], dtype=torch.long)  # 标签也是整数类型
        # print('getitem.size:',data.size(),label.size())
        return data, label

    def _build_voca_list(self):
        # 创建词表，并将词表保存和输出
        voca_str = "abcdefghijklmnopqrstuvwxyz"
        voca_list = {
            "pad": 0,
        }
        index = 1
        for char in voca_str:
            voca_list[char] = index
            index += 1
        voca_list["unk"] = index
        #  将词表写入对应文件
        with open(self.file_name, "w", encoding="utf8") as file:
            json.dump(voca_list, file, ensure_ascii=False, indent=2)

        return index + 1, voca_list

    def _build_samples(self):
        """
        创建样本用例，如果初始样本包含目标字符，则无操作，否则按照一定概率向样例中增加一定数量的目标字符
        返回值：
            output_data 样例文本数据
            output_label 样例对应标签
            voca_list 词表文件
            max_index_len 最大样例长度
            voc_max_len 词表长度
        """
        voc_max_len, voca_list = self._build_voca_list()
        cand_str = "abcdefghijklmnopqrstuvwxyz"
        set_seed(self.seed)
        output_data = []
        output_label = []
        max_index_len = 0
        for num in range(self.num_samples):
            curr_sample_length = random.randint(
                self.sample_length // 2, self.sample_length * 3
            )
            random_index = [
                random.randint(1, voc_max_len - 2) for _ in range(curr_sample_length)
            ]
            data = "".join([cand_str[index - 1] for index in random_index])
            if max_index_len < len(random_index):
                max_index_len = len(random_index)
            label = []
            # 如果目标字符不在初始样例中，则按照一定概率增加
            if target_index not in random_index:
                flag = random.randint(0, curr_sample_length - 1)
                if flag >= curr_sample_length // 2:
                    curr_range = random.randint(0, curr_sample_length // 2)
                    idx = [
                        random.randint(0, curr_sample_length - 1)
                        for _ in range(curr_range)
                    ]
                    for i in idx:
                        random_index[i] = target_index
            # 根据样例创建label
            for idx in random_index:
                if idx == target_index:
                    label.append(1)
                else:
                    label.append(0)
            output_data.append(data)
            output_label.append(label)
        set_seed()
        # print('build samples:',len(output_data),len(output_label))
        # print(output_data)
        return output_data, output_label, voca_list, max_index_len, voc_max_len

    def _to_sequence(self):
        # 将样例转换为序列
        sequence_index = []
        for token in self.data:
            sequence_index.append([self.voca_list[word] for word in token])
        return sequence_index

    def _padding(self):
        # 统一样例长度
        sequence_index = self._to_sequence()
        res = []
        res_label = []
        for index, label in zip(sequence_index, self.label):
            if len(index) > self.max_len:
                index = index[: self.max_len]
            else:
                index += [0] * (self.max_len - len(index))
                label += [0] * (self.max_len - len(label))
            res.append(index)
            res_label.append(label)
        # print('padding:', len(res),len(res_label))
        return res, res_label

    def _return_length(self):
        # 返回最大样例长度
        return self.max_len, self.voca_max_len


class Locate_index(nn.Module):
    """
    定义目标字符所在位置的模型，模型输出对应位置为1，则表示为目标字符，否则非目标字符
    hidden_size 隐藏层节点数
    output_size 输出维度
    embedding_dims 嵌入层输出和rnn输入， 单个样例需要的维度
    num_embeddings 嵌入层输入，和词表长度相同，
    num_layers rnn层数
    dropout rnn网络dropout参数
    target_index 目标字符对应词表的索引
    """

    def __init__(
        self,
        hidden_size,
        output_size,
        embedding_dims,
        num_embeddings,
        num_layers,
        dropout,
        target_index,
    ):
        # 定义网络模型和损失函数
        super(Locate_index, self).__init__()
        self.target_index = target_index
        self.embedding_layer = nn.Embedding(num_embeddings, embedding_dims)
        self.rnn_layers = nn.RNN(
            embedding_dims,
            hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
        )
        # self.rnn_layers = nn.RNN(input_size, hidden_size, batch_first=True)
        self.Linlayers1 = nn.Linear(hidden_size, 3 * hidden_size)
        self.Linlayers2 = nn.Linear(3 * hidden_size, 3 * hidden_size)
        self.Linlayers3 = nn.Linear(3 * hidden_size, 2 * hidden_size)
        self.Linlayers4 = nn.Linear(2 * hidden_size, 3 * hidden_size)
        self.Linlayers5 = nn.Linear(3 * hidden_size, 4 * hidden_size)
        self.Linlayers6 = nn.Linear(4 * hidden_size, hidden_size)
        self.classifier = nn.Linear(hidden_size, output_size)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, y=None):
        # 定义前向传播
        # print("x.size",x.size())
        x = self.embedding_layer(x)
        rnn_output, hidden_output = self.rnn_layers(x)
        # 这里取整个rnn的输出作为分类网络的输入
        y_pred = self.classifier(
            self.Linlayers6(
                self.Linlayers5(
                    self.Linlayers4(
                        self.Linlayers3(self.Linlayers2(self.Linlayers1(rnn_output)))
                    )
                )
            )
        )
        pred_index = torch.argmax(torch.softmax(y_pred, dim=-1), dim=-1)
        y_pred = y_pred.view(
            -1, y_pred.size(-1)
        )  # [batch_size * sequence_length, num_classes]
        # print('y_pred_size:',y_pred.size())
        # print('pred_index.size:',pred_index.size())
        if y is not None:
            # 调整 y 的形状
            y = y.view(-1)  # [batch_size * sequence_length]
            # print('y.size:',y.size())
            loss = self.loss(y_pred, y)
            return loss, pred_index
        else:
            return pred_index


class TrainAndPredict(nn.Module):
    """
    定义训练和预测类
    输入参数：
        target_size 目标字符在词表中的索引
        num_samples 样例数量
        sample_length  单个样例默认大小
        embedding_dims  词表单个字符维度
        voca_name  词表文件名称
        epoches 总的训练代数
        start_epoch 开始训练的代数，断点训练用
        hidden_size 隐藏层节点数
        output_size 输出层节点数
        num_embeddings 词表的长度
    """

    def __init__(
        self,
        target_index,
        num_samples,
        sample_length,
        embedding_dims,
        voca_name,
        epoches,
        start_epoch,
        hidden_size,
        output_size,
        num_embeddings,
    ):
        super(TrainAndPredict, self).__init__()
        self.target_index = target_index
        os.makedirs("./checkpoints/", exist_ok=True)
        self.root_dir = "./checkpoints"
        self.num_samples = num_samples
        self.sample_length = sample_length
        self.embedding_dims = embedding_dims
        self.voca_name = voca_name
        self.num_embeddings = num_embeddings
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.end_epoch = epoches
        self.start_epoch = start_epoch
        self.lr = 1e-5
        self.num_layers = 20
        self.dropout = 0.5
        self.epoch = 0
        self.flag = 0
        self.best_loss = float("inf")
        # 定义rnn_model，optim以及自动更新学习率，以及signal信号处理
        self.rnn_model = Locate_index(
            hidden_size=hidden_size,
            output_size=self.output_size,
            embedding_dims=self.embedding_dims,
            num_embeddings=self.num_embeddings,
            num_layers=self.num_layers,
            dropout=self.dropout,
            target_index=self.target_index,
        ).to(device)
        self.rnn_model_optimizer = torch.optim.Adam(
            self.rnn_model.parameters(), lr=self.lr, weight_decay=1e-5
        )
        self.rnn_scheduler = ReduceLROnPlateau(self.rnn_model_optimizer, "min", 0.1, 50)
        self.best_para = self.rnn_model.state_dict()
        self.acc_log = []
        self.loss_log = []
        signal.signal(signal.SIGINT, self.signal_handler)

    def train_data(self, dataloader):
        # 定义训练函数，传入dataloader包含train_dataloader和test_dataloader
        epoch_len = len(str(self.end_epoch))

        train_dataloader, test_dataloader = dataloader
        try:
            for epoch in range(self.start_epoch, self.end_epoch):
                start_time = time.time()
                curr_loss = 0
                self.flag += 1
                f1, recall = 0, 0
                self.rnn_model.train()
                for data in train_dataloader:
                    input, label = data

                    input = input.to(device)
                    label = label.to(device)
                    self.rnn_model_optimizer.zero_grad()
                    tmp_loss, y_pred = self.rnn_model(input, label)
                    curr_loss += tmp_loss.item() / len(dataloader)
                    tmp_loss.backward()
                    if isinstance(y_pred, torch.Tensor):
                        y_pred = y_pred.cpu().numpy()
                    if isinstance(label, torch.Tensor):
                        label = label.cpu().numpy()

                    f1 += f1_score(
                        label.flatten(),
                        y_pred.flatten(),
                        average="macro",
                        zero_division=0,
                    )
                    recall += recall_score(
                        label.flatten(),
                        y_pred.flatten(),
                        average="macro",
                        zero_division=0,
                    )
                    self.rnn_model_optimizer.step()
                self.epoch = epoch + 1
                f1, recall = f1 / len(train_dataloader), recall / len(train_dataloader)
                self.loss_log.append(curr_loss)
                if curr_loss <= self.best_loss:
                    self.best_loss = curr_loss
                    self.best_para = self.rnn_model.state_dict()
                    self._save_checkpoints()
                    self.flag = 0
                if self.flag > 10000 or epoch == self.end_epoch:
                    self._save_checkpoints(file_name="final_checkpoints.pth.tar")
                    break

                correct = 0
                total = 0
                total_f1, total_recall = 0, 0
                num_batches = len(test_dataloader)
                real_length = 0
                for data in test_dataloader:
                    real_length += 1
                    inputs, labels = data
                    inputs = inputs.to(device)
                    self.rnn_model.eval()
                    with torch.no_grad():
                        preds = self.rnn_model(inputs)
                        preds = (
                            preds.cpu().numpy()
                            if isinstance(preds, torch.Tensor)
                            else preds
                        )
                        labels = (
                            labels.numpy()
                            if isinstance(labels, torch.Tensor)
                            else labels
                        )
                        total_f1 += f1_score(
                            labels.flatten(),
                            preds.flatten(),
                            average="macro",
                            zero_division=0,
                        )
                        total_recall += recall_score(
                            labels.flatten(),
                            preds.flatten(),
                            average="macro",
                            zero_division=0,
                        )
                        comparison = np.equal(preds, labels)
                        # print(comparison)
                        tmp_correct = np.sum(comparison)
                        tmp_total = preds.size
                        correct += tmp_correct
                        total += tmp_total
                # 计算所有批次的平均F1分数和召回率
                test_f1 = total_f1 / num_batches
                test_rec = total_recall / num_batches
                # 计算总体准确率
                acc = correct / total
                self.acc_log.append(acc)
                dur_time = time.time() - start_time
                print(
                    f"第{epoch + 1:{epoch_len}d}轮, curr_loss:{curr_loss:.3f}, best_loss:{self.best_loss:.3f},train_f1_score:{f1:.3f},train_recall:{recall:.3f}, test_acc{acc:.3f}, test_f1_score:{test_f1:.3f}, test_recall:{test_rec:.3f}, dur_time:{dur_time:.2f}"
                )
                # lr自动更新
                self.rnn_scheduler.step(curr_loss)
        except Exception as e:
            self._save_checkpoints()
            print(f"error in train {e}")

    def predict_value(self, pred_dataloader):
        # 定义训练函数
        try:
            start_time = time.time()
            self.rnn_model.eval()
            checkpoints = torch.load(
                os.path.join(self.root_dir, "final_checkpoints.pth.tar")
            )
            self.rnn_model.load_state_dict(checkpoints["state_dict"])
            correct = 0
            total = 0
            curr_f1, curr_recall = 0, 0
            num_batches = len(test_dataloader)
            for data in test_dataloader:
                inputs, labels = data
                inputs = inputs.to(device)
                self.rnn_model.eval()
                with torch.no_grad():
                    preds = self.rnn_model(inputs)
                    preds = (
                        preds.cpu().numpy()
                        if isinstance(preds, torch.Tensor)
                        else preds
                    )
                    labels = (
                        labels.numpy() if isinstance(labels, torch.Tensor) else labels
                    )
                    curr_f1 += f1_score(
                        labels.flatten(),
                        preds.flatten(),
                        average="macro",
                        zero_division=0,
                    )
                    curr_recall += recall_score(
                        labels.flatten(),
                        preds.flatten(),
                        average="macro",
                        zero_division=0,
                    )
                    comparison = np.equal(preds, labels)
                    # print(comparison)
                    tmp_correct = np.sum(comparison)
                    tmp_total = preds.size
                    correct += tmp_correct
                    total += tmp_total
            # 计算所有批次的平均F1分数和召回率
            eval_f1 = curr_f1 / num_batches
            eval_rec = curr_recall / num_batches
            # 计算总体准确率
            acc = correct / total
            self.acc_log.append(acc)
            dur_time = time.time() - start_time
            print(
                f"eval_acc{acc:.3f}, eval_f1_score:{eval_f1:.3f}, eval_recall:{eval_rec:.3f}, dur_time:{dur_time:.2f}"
            )
        except Exception as e:
            print(f"error in prdict {e}")

    def signal_handler(self, signal, frame):
        self._save_checkpoints()

    def _load_weights(self, checkpoint_path):
        # 定义权重加载函数
        checkpoints = torch.load(checkpoint_path, map_location=device)
        self.rnn_model.load_state_dict(checkpoints["state_dict"])
        self.start_epoch = checkpoints["epoch"]
        self.best_loss = checkpoints["loss"]
        self.rnn_model_optimizer.load_state_dict(checkpoints["optimizer"])
        self.rnn_scheduler.load_state_dict(checkpoints["schedule"])
        self.flag = checkpoints["flag"]

    def _save_checkpoints(self, file_name=None):
        # 定义权重保存
        try:
            checkpoints = {
                "epoch": self.epoch,
                "state_dict": self.best_para,
                "loss": self.best_loss,
                "optimizer": self.rnn_model_optimizer.state_dict(),
                "schedule": self.rnn_scheduler.state_dict(),
                "flag": self.flag,
            }
            if file_name is None:
                file_name = "best_checkpoint.pth.tar"
            file_name = os.path.join(self.root_dir, file_name)
            torch.save(checkpoints, file_name)
            print(f"checkpoint save {file_name}")
        except Exception as e:
            print(f"error in save checkpoints {e}")


def collate_fn(batch):
    # batch 是一个列表，包含了 (data, label)
    data_list, label_list = zip(*batch)
    # 对数据和标签进行填充
    data_list = [data.clone().detach() for data in data_list]
    label_list = [label.clone().detach() for label in label_list]
    padded_data = pad_sequence(data_list, batch_first=True, padding_value=0)
    padded_labels = pad_sequence(label_list, batch_first=True, padding_value=0)

    return padded_data, padded_labels


if __name__ == "__main__":
    target_index = 5
    num_samples = 100000
    sample_length = 20
    embedding_dims = 15 if 15 >= 3 * sample_length else 3 * sample_length + 3
    file_name = "voca.json"
    epoches = 100000
    hidden_size = 150
    output_size = 2
    generate_dataset = Random_Dataset(
        target_index=target_index,
        num_samples=num_samples,
        sample_length=sample_length,
        embedding_dims=embedding_dims,
        file_name=file_name,
        seed=10101,
    )
    train_data_len = int(len(generate_dataset) * 0.9)
    test_data_len = len(generate_dataset) - train_data_len
    train_dataset, test_dataset = random_split(
        generate_dataset, [train_data_len, test_data_len]
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=256,
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=True,
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=256, collate_fn=collate_fn, drop_last=True
    )
    data_loader = [train_dataloader, test_dataloader]
    sample_length, num_embeddings = generate_dataset._return_length()

    TrainOrPredict = TrainAndPredict(
        target_index,
        num_samples,
        sample_length,
        embedding_dims,
        file_name,
        epoches,
        0,
        hidden_size,
        output_size,
        num_embeddings,
    )
    if os.listdir('.\checkpoints'):
        checkpoint_path = r'checkpoints\best_checkpoint.pth.tar'
        TrainOrPredict._load_weights(checkpoint_path)
    
    TrainOrPredict.train_data(data_loader)
    pred_dataset = Random_Dataset(
        target_index=target_index,
        num_samples=num_samples,
        sample_length=sample_length,
        embedding_dims=embedding_dims,
        file_name=file_name,
        seed=101010,
    )
    pred_data_loader = DataLoader(
        pred_dataset, batch_size=256, collate_fn=collate_fn, drop_last=True
    )
    TrainOrPredict.predict_value(pred_data_loader)
