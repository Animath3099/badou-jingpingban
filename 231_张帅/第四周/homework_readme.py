#week3作业
import numpy as np
#词典；每个词后方存储的是其词频，词频仅为示例，不会用到，也可自行修改
Dict = {"经常":0.1,
        "经":0.05,
        "有":0.1,
        "常":0.001,
        "有意见":0.1,
        "歧":0.001,
        "意见":0.2,
        "分歧":0.2,
        "见":0.05,
        "意":0.05,
        "见分歧":0.05,
        "分":0.1}

#待切分文本
sentence = "经常有意见分歧"

#实现全切分函数，输出根据字典能够切分出的所有的切分方式
def all_cut(sentence, Dict):
    # 递归头
    if len(sentence) == 1:
        return [[sentence]]
    elif len(sentence) == 0:
        return []

    res = []
    start_list = []
    words_list = list(Dict.keys())
    # 将文本开头所有可能切分出来
    ([start_list.append([word]) for word in words_list if sentence.startswith(word)])
    # 将每种开头都循环一次
    for start in start_list:
        split = sentence.split(start[0],maxsplit=2)
        end_list = all_cut(split[1], Dict)
        if len(end_list) == 0:
            res.append(start)
        for end in end_list:
            cur = []
            cur.extend(start)
            cur.extend(end)
            res.append(cur)
    return res

all = all_cut(sentence,Dict)
[print(one) for one in all]

#目标输出;顺序不重要
target = [
    ['经常', '有意见', '分歧'],
    ['经常', '有意见', '分', '歧'],
    ['经常', '有', '意见', '分歧'],
    ['经常', '有', '意见', '分', '歧'],
    ['经常', '有', '意', '见分歧'],
    ['经常', '有', '意', '见', '分歧'],
    ['经常', '有', '意', '见', '分', '歧'],
    ['经', '常', '有意见', '分歧'],
    ['经', '常', '有意见', '分', '歧'],
    ['经', '常', '有', '意见', '分歧'],
    ['经', '常', '有', '意见', '分', '歧'],
    ['经', '常', '有', '意', '见分歧'],
    ['经', '常', '有', '意', '见', '分歧'],
    ['经', '常', '有', '意', '见', '分', '歧']
]
