# 导入copy模块，使得后续可以进行列表的深拷贝
import copy

# 词典，包含词及其词频（这里词频没有被使用）
Dict = {
    "经常": 0.1,
    "经": 0.05,
    "有": 0.1,
    "常": 0.001,
    "有意见": 0.1,
    "歧": 0.001,
    "意见": 0.2,
    "分歧": 0.2,
    "见": 0.05,
    "意": 0.05,
    "见分歧": 0.05,
    "分": 0.1
}

# 待切分的文本
sentence = "经常有意见分歧"


# 全切分函数
def full_cut(sentence, dict_keys, max_word_length):
    # 如果句子为空，则返回包含空列表的列表
    if not sentence:
        return [[]]

    # 初始化一个列表来存储所有切分的可能
    cuts = []

    # 尝试每一个可能的单词长度
    for i in range(1, min(max_word_length, len(sentence)) + 1):
        # 如果当前前缀是词典中的词
        if sentence[:i] in dict_keys:
            # 递归进行剩余部分的切分
            for sub_cut in full_cut(sentence[i:], dict_keys, max_word_length):
                # 将当前词和剩余部分的切分结果组合起来
                cuts.append([sentence[:i]] + sub_cut)

    # 返回所有切分的可能
    return cuts
# # 全切分函数
# def full_cut(sentence, Dict):
#     # 所有可能的切分方法
#     results = []
#     # 递归切分函数
#     def _cut(s, path):
#         # 如果句子已经为空，则将当前路径加入结果中
#         if not s:
#             results.append(path)
#             return
#         # 遍历词典，尝试每一个可能的词
#         for word in Dict.keys():
#             if s.startswith(word):
#                 # 如果句子以词典中的词开头，则递归切分剩余的句子
#                 _cut(s[len(word):], path + [word])
#     # 初始化递归切分
#     _cut(sentence, [])
#     return results
#
# # 输出所有可能的切分方式
# for cut in full_cut(sentence, Dict):
#     print(cut)

# 获取词典中最长的词的长度
max_word_length = max(map(len, Dict.keys()))

# 输出所有可能的切分方式
all_possible_cuts = full_cut(sentence, set(Dict.keys()), max_word_length)
for cut in all_possible_cuts:
    print(cut)
