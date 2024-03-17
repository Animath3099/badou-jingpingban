import numpy as np
import random
import copy

'''
维特比解码和beam search
    这个脚本的主要目的是展示两种不同的序列解码算法：维特比算法和束搜索算法。
    这两种算法都可以找出在一个给定的网格中的最优路径。
    维特比算法保证找到最优解，而束搜索算法通过限制搜索宽度（beam_size）来减少计算量，可能找到次优的解。
    这个脚本通过对比两种算法的输出，可以帮助理解它们在寻找最优路径时的不同表现。
    在实际应用中，这些算法可以用于各种序列模型的解码过程，如隐马尔可夫模型（HMM）或条件随机场（CRF）。
'''


# 定义了一个名为Fence的类，它代表一个由节点组成的网格，每个节点可以通过其行列索引来标识
class Fence:
    # 类的初始化方法，n代表网格的宽度（列数），h代表网格的高度（行数）
    def __init__(self, n, h):
        # 初始化网格的宽度和高度
        self.width = n
        self.height = h

    # 用行列组成的list代表一个节点，每两个相邻的列的节点之间可以计算距离
    # e.g:node1 = [2,1] node2 = [3, 2]
    # 为两个节点给一个固定的路径分值
    def score(self, node1, node2):  # 定义了一个方法score，用以计算两个相邻节点之间的路径分值
        if node1 == "start":  # 如果第一个节点是起始节点（字符串"start"），则计算并返回一个基于第二个节点行列索引的分数
            return (node2[0] + node2[1] + 1) / (node2[0] * node2[1] + 1)
        assert node1[1] + 1 == node2[1]  # 保证两个节点处于相邻列:确保两个节点处于相邻列，否则抛出异常
        # 计算两个节点之间的路径分值，返回计算后的分数
        mod = (node1[0] + node1[1] + node2[0] + node2[1]) % 3 + 1
        mod /= node1[0] * 4 + node1[1] * 3 + node2[0] * 2 + node2[1] * 1
        return mod


class Path:  # 定义了一个名为Path的类，代表一条由多个节点组成的路径，并记录该路径的总分
    # 定义一个路径
    # 路径由数个节点组成，并且具有一个路径总分
    def __init__(self):
        # 初始化路径，起始只有一个“start”节点，分数为0
        self.nodes = ["start"]
        self.score = 0

    def __len__(self):
        # 定义了一个__len__方法，返回路径中节点的数量，使得Path对象可以使用len()函数
        return len(self.nodes)


def beam_search(fence, beam_size):  # 定义了一个名为beam_search的函数，它实现了束搜索算法
    # 获取网格的宽度和高度
    width = fence.width
    height = fence.height
    # 创建一个初始路径，并将其放入束缓冲区中
    starter = Path()
    beam_buffer = [starter]
    # 初始化一个新的束缓冲区，用于存储扩展后的路径
    new_beam_buffer = []
    while True:  # 开始一个循环，直到找到所有列的最优路径
        for path in beam_buffer:  # 遍历当前束缓冲区中的每条路径
            path_length = len(path) - 1  # 计算当前路径的长度（减1是因为包含了"start"节点）
            for h in range(height):  # 遍历每一行，尝试将新节点添加到路径中
                node = [h, path_length]  # 创建一个新节点，行索引为h，列索引为当前路径的长度
                new_path = copy.deepcopy(path)  # 深度复制当前路径，以便添加新节点
                new_path.score += fence.score(path.nodes[-1], node)  # 更新新路径的分数，并添加新节点
                new_path.nodes.append(node)
                new_beam_buffer.append(new_path)  # 将新路径添加到新的束缓冲区中
        new_beam_buffer = sorted(new_beam_buffer, key=lambda x: x.score)  # 将新的束缓冲区中的路径按照分数排序
        beam_buffer = new_beam_buffer[:beam_size]  # 将分数最高的beam_size条路径保留在当前束缓冲区中
        new_beam_buffer = []  # 清空新的束缓冲区，为下一轮扩展做准备
        if len(beam_buffer[0]) == width + 1:  # 如果当前束缓冲区中的路径长度达到网格的宽度加1（包含"start"节点），则结束循环
            break
    return beam_buffer  # 返回束缓冲区中的路径，即束搜索的结果


def viterbi(fence):  # 定义了一个名为viterbi的函数，它实现了维特比算法====viterbi函数的结构与beam_search相似，但在路径扩展和选择方式上有所不同
    # 定义一个网格的宽度和高度，创建Fence实例
    width = fence.width
    height = fence.height
    starter = Path()
    beam_buffer = [starter]
    new_beam_buffer = []
    while True:
        for h in range(height):
            path_length = len(beam_buffer[0]) - 1
            node = [h, path_length]
            node_path = []
            for path in beam_buffer:
                new_path = copy.deepcopy(path)
                new_path.score += fence.score(path.nodes[-1], node)
                new_path.nodes.append(node)
                node_path.append(new_path)
            node_path = sorted(node_path, key=lambda x: x.score)
            new_beam_buffer.append(node_path[0])
        beam_buffer = new_beam_buffer
        new_beam_buffer = []
        if len(beam_buffer[0]) == width + 1:
            break
    return sorted(beam_buffer, key=lambda x: x.score)


# 定义一个网格的宽度和高度，创建Fence实例
width = 6
height = 4
fence = Fence(width, height)
# print(fence.score([1,2], [3,3]))
# 执行束搜索算法，并打印出每条路径的节点和分数
beam_size = 1
res = beam_search(fence, beam_size)
for i in range(beam_size):
    print(res[i].nodes, res[i].score)
print("-----------")
# 执行维特比算法，并打印出每条路径的节点和分数
res = viterbi(fence)
for path in res:
    print(path.nodes, path.score)
