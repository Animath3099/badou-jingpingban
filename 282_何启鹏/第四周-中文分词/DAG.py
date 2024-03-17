#week3作业

#词典；每个词后方存储的是其词频，词频仅为示例，不会用到，也可自行修改
def build_dag(pdict, sentence):
    DAG = {}
    N = len(sentence)
    for i in range(N):
        tmp_list = []
        j = i + 1
        subword = sentence[i:j]
        while j <= N :
            if subword in pdict:
                tmp_list.append(j-1)
            j += 1
            subword = sentence[i:j]
        DAG[i] = tmp_list
    return DAG

sentence = "经常有意见分歧"
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
sen_list = list(Dict)
dag = build_dag(Dict, sentence)
print(dag)



def get_all_possible_words(DAG, sentence):
    # 获取句子长度
    N = len(sentence)
    # 定义初始栈，包含起始位置 0 和空路径
    stack = [(0, [])]
    # 定义结果列表
    results = []
    # 当栈不为空时，持续循环
    while stack:
        # 取出栈顶元素，包括当前位置和路径
        pos, path = stack.pop()
        # 如果当前位置已经到达句子末尾，将当前路径加入结果列表
        if pos == N:
            results.append(path)
            print(path)
        # 否则，对当前位置的所有后继节点进行遍历
        else:
            for next_pos in DAG[pos]:
                # 计算当前位置和后继节点之间的词语
                word = sentence[pos:next_pos+1]
                # 将新词语添加到路径末尾，得到新路径
                new_path = path + [word]
                # 将新路径和后继节点位置入栈，等待下一轮遍历
                stack.append((next_pos+1, new_path))
    # 返回所有可能的切分组合
    return results
res = get_all_possible_words(dag, sentence)
print(len(res))
