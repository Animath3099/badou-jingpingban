# 定义一个字典，存储词汇及其对应的词频
word_dict = {"经常": 0.1,
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
             "分": 0.1}

# 定义待切分的文本
text = "经常有意见分歧"

# 定义回溯函数backtrack
def backtrack(start, path, text, word_dict, result):
    # 如果已经到达文本结尾，则将当前路径添加到结果中
    if start == len(text):
        result.append(path[:])
        return

    # 尝试从起始位置到结尾位置的每个子串
    for end in range(start + 1, len(text) + 1):
        word = text[start:end]
        # 如果该子串在字典中存在，则将其添加到路径中，并递归处理剩余部分
        if word in word_dict:
            path.append(word)
            backtrack(end, path, text, word_dict, result)
            path.pop()

# 定义函数all_cut，根据给定字典能够切分出所有的切分方式
def all_cut(text, word_dict):
    # 定义结果列表
    result = []

    # 从文本开头开始尝试
    backtrack(0, [], text, word_dict, result)

    # 返回结果列表
    return result

# 调用函数all_cut，输出所有根据字典能够切分出的切分方式
result = all_cut(text, word_dict)

# 输出结果
print(result)

# 输出结果如下
# [['经', '常', '有', '意', '见', '分', '歧'],
#  ['经', '常', '有', '意', '见', '分歧'],
#  ['经', '常', '有', '意', '见分歧'],
#  ['经', '常', '有', '意见', '分', '歧'],
#  ['经', '常', '有', '意见', '分歧'],
#  ['经', '常', '有意见', '分', '歧'],
#  ['经', '常', '有意见', '分歧'],
#  ['经常', '有', '意', '见', '分', '歧'],
#  ['经常', '有', '意', '见', '分歧'],
#  ['经常', '有', '意', '见分歧'],
#  ['经常', '有', '意见', '分', '歧'],
#  ['经常', '有', '意见', '分歧'],
#  ['经常', '有意见', '分', '歧'],
#  ['经常', '有意见', '分歧']]
