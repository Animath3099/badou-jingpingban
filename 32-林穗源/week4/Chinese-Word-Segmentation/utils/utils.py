import logging
import os
import matplotlib.pyplot as plt

def log_display(train_log):
    """
    训练日志画图
    :param train_log: 训练日志
    :return:
    """
    plt.xlabel('epoch')
    plt.plot(range(len(train_log)), [l[1] for l in train_log], label='loss')  # 画loss曲线
    plt.plot(range(len(train_log)), [l[0] for l in train_log], label='acc')  # 画acc曲线
    plt.legend()
    plt.savefig('train.jpg')
    plt.show()
def logger_config(log_path,logging_name):
    """
    配置log
    :param log_path: 输出log路径
    :param logging_name: 记录中name，可随意
    :return:
    """
    # 获取logger对象,取名
    logger = logging.getLogger(logging_name)
    # 输出DEBUG及以上级别的信息，针对所有输出的第一层过滤
    logger.setLevel(level=logging.DEBUG)
    # 获取文件日志句柄并设置日志级别，第二层过滤
    handler = logging.FileHandler(log_path, encoding='UTF-8')
    handler.setLevel(logging.INFO)
    # 生成并设置文件日志格式
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    # console相当于控制台输出，handler文件输出。获取流句柄并设置日志级别，第二层过滤
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    # 为logger对象添加句柄
    logger.addHandler(handler)
    logger.addHandler(console)
    return logger

log_path = 'train.log'
if os.path.exists(log_path):
    os.remove(log_path)
logger = logger_config(log_path=log_path, logging_name='train-log')