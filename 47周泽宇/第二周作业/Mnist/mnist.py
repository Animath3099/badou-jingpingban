# 此模块用于下载载入MNIST数据集

import gzip
import pickle
import os
import numpy as np
import urllib.request

# 下载使用的地址以及下载文件字典
url_base = "http://yann.lecun.com/exdb/mnist/"
key_file = {
    "train_img": "train-images-idx3-ubyte.gz",
    "train_label": "train-labels-idx1-ubyte.gz",
    "test_img": "t10k-images-idx3-ubyte.gz",
    "test_label": "t10k-labels-idx1-ubyte.gz",
}

# 获取当前目录与即将创建的mnist.pkl文件路径
dataset_dir = os.path.dirname(os.path.abspath(__file__))
save_file = dataset_dir + "/mnist.pkl"

# 声明一些变量存储数据维度等信息
train_num = 60000
test_num = 10000
img_dim = (1, 28, 28)
img_size = 784

# 按照文件名下载 - 私有函数
def _dowload(file_name):
    file_path = dataset_dir + "/" + file_name

    if os.path.exists(file_path):
        return

    print("正在下载 " + file_name + " ... ", end="\t\t\t")
    urllib.request.urlretrieve(url_base + file_name, file_path)
    print("OK")


# 下载整个列表
def download_mnist():
    for file in key_file.values():
        _dowload(file)


# 按照文件名通过下载的gzip文件导入标签
def _load_label(file_name):
    file_path = dataset_dir + "/" + file_name

    print("正在转换 " + file_name + " 到NumPy数组 ...", end="\t\t\t")
    with gzip.open(file_path, "rb") as file:
        labels = np.frombuffer(file.read(), np.uint8, offset=8)
    print("OK")

    return labels


# 按照文件名通过下载的gzip文件导入图片数据
def _load_img(file_name):
    file_path = dataset_dir + "/" + file_name

    print("正在转换 " + file_name + " 到NumPy数组 ...", end="\t\t\t")
    with gzip.open(file_path, "rb") as file:
        data = np.frombuffer(file.read(), np.uint8, offset=16)
    data = data.reshape(-1, img_size)
    print("OK")

    return data


# 加载整个列表到内存
def _convert_numpy():
    dataset = {}
    dataset["train_img"] = _load_img(key_file["train_img"])
    dataset["train_label"] = _load_label(key_file["train_label"])
    dataset["test_img"] = _load_img(key_file["test_img"])
    dataset["test_label"] = _load_label(key_file["test_label"])

    return dataset


# 初始化数据集
def init_mnist():
    # download_mnist()
    dataset = _convert_numpy()
    print("创建pkl保存文件 ...", end="\t\t\t")
    with open(save_file, "wb") as file:
        pickle.dump(dataset, file, -1)
    print("OK")


# 将正确解标签转换为one_hot形式的函数
def _change_one_hot_label(X):
    T = np.zeros((X.size, 10))
    for idx, row in enumerate(T):
        row[X[idx]] = 1

    return T


# 加载使用数据集
def load_mnist(normalize=True, flatten=True, one_hot_label=False):
    """读入MNIST数据集

    参数:
    ----------
    normalize : 是否将图像的像素值正规化为0.0~1.0
    one_hot_label :
        one_hot_label为True的情况下,标签作为one-hot数组返回
        one-hot数组是指[0,0,1,0,0,0,0,0,0,0]这样的数组
    flatten : 是否将图片展开成一维数组

    返回值:
    ----------
    (训练图像, 训练标签), (测试图像, 测试标签)
    """
    if not os.path.exists(save_file):
        init_mnist()

    with open(save_file, "rb") as file:
        dataset = pickle.load(file)

    if normalize:
        for key in ("train_img", "test_img"):
            dataset[key] = dataset[key].astype(np.float32)
            dataset[key] /= 255.0

    # if one_hot_label:
    #     dataset["train_label"] = _change_one_hot_label(dataset["train_label"])
    #     dataset["test_label"] = _change_one_hot_label(dataset["test_label"])

    if not flatten:
        for key in ("train_img", "test_img"):
            dataset[key] = dataset[key].reshape(-1, 1, 28, 28)

    return (dataset["train_img"], dataset["train_label"]), (
        dataset["test_img"],
        dataset["test_label"],
    )

