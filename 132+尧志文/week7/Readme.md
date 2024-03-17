# 测试区分好评or差评
# config里面num_time参数为测试多少条数据花费时间的参数
# Bert
Config = {
    "model_path": "output",
    "train_data_path": "train_data.json",
    "valid_data_path": "valid_data.json",
    "vocab_path":"chars.txt",
    "model_type":"bert",
    "max_length": 30,
    "hidden_size": 256,
    "kernel_size": 3,
    "num_layers": 2,
    "epoch": 15,
    "batch_size": 128,
    "pooling_style":"max",
    "optimizer": "adam",
    "learning_rate": 1e-3,
    "pretrain_model_path":r"E:\models\bert-base-chinese",
    "seed": 987,
    "num_time": 100
}

以上参数下，准确率 0.87， 100条数据平均执行时间0.04383626秒
# CNN
Config = {
    "model_path": "output",
    "train_data_path": "train_data.json",
    "valid_data_path": "valid_data.json",
    "vocab_path":"chars.txt",
    "model_type":"bert",
    "max_length": 30,
    "hidden_size": 256,
    "kernel_size": 3,
    "num_layers": 2,
    "epoch": 15,
    "batch_size": 128,
    "pooling_style":"max",
    "optimizer": "adam",
    "learning_rate": 1e-3,
    "pretrain_model_path":r"E:\models\bert-base-chinese",
    "seed": 987,
    "num_time": 100
}
以上参数下，准确率 0.88， 100条数据平均执行时间0.00159135秒


# fast_text模型
Config = {
    "model_path": "output",
    "train_data_path": "train_data.json",
    "valid_data_path": "valid_data.json",
    "vocab_path":"chars.txt",
    "model_type":"fast_text",
    "max_length": 30,
    "hidden_size": 256,
    "kernel_size": 3,
    "num_layers": 2,
    "epoch": 15,
    "batch_size": 128,
    "pooling_style":"max",
    "optimizer": "adam",
    "learning_rate": 1e-3,
    "pretrain_model_path":r"E:\models\bert-base-chinese",
    "seed": 987,
    "num_time": 100
}

以上参数下，准确率 0.88， 100条数据平均执行时间0.00050998秒

# LSTM
Config = {
    "model_path": "output",
    "train_data_path": "train_data.json",
    "valid_data_path": "valid_data.json",
    "vocab_path":"chars.txt",
    "model_type":"lstm",
    "max_length": 30,
    "hidden_size": 256,
    "kernel_size": 3,
    "num_layers": 2,
    "epoch": 15,
    "batch_size": 128,
    "pooling_style":"max",
    "optimizer": "adam",
    "learning_rate": 1e-3,
    "pretrain_model_path":r"E:\models\bert-base-chinese",
    "seed": 987,
    "num_time": 100
}

以上参数下，准确率 0.874479， 100条数据平均执行时间0.00660638秒

# GRU
Config = {
    "model_path": "output",
    "train_data_path": "train_data.json",
    "valid_data_path": "valid_data.json",
    "vocab_path":"chars.txt",
    "model_type":"gru",
    "max_length": 30,
    "hidden_size": 256,
    "kernel_size": 3,
    "num_layers": 2,
    "epoch": 15,
    "batch_size": 128,
    "pooling_style":"max",
    "optimizer": "adam",
    "learning_rate": 1e-3,
    "pretrain_model_path":r"E:\models\bert-base-chinese",
    "seed": 987,
    "num_time": 100
}
以上参数下，准确率 0.862385， 100条数据平均执行时间0.00566481秒

# RNN
Config = {
    "model_path": "output",
    "train_data_path": "train_data.json",
    "valid_data_path": "valid_data.json",
    "vocab_path":"chars.txt",
    "model_type":"rnn",
    "max_length": 30,
    "hidden_size": 256,
    "kernel_size": 3,
    "num_layers": 2,
    "epoch": 15,
    "batch_size": 128,
    "pooling_style":"max",
    "optimizer": "adam",
    "learning_rate": 1e-3,
    "pretrain_model_path":r"E:\models\bert-base-chinese",
    "seed": 987,
    "num_time": 100
}

以上参数下，准确率 0.864053， 100条数据平均执行时间0.00332780秒


