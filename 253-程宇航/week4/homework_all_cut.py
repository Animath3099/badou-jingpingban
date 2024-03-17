#week3作业
#实现全切分函数，输出根据字典能够切分出的所有的切分方式

"""照搬了几位同学代码，直呼震惊，以示敬意"""

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

# 第一种方法    递归
"""
def all_cut(sentence, Dict):
    if len(sentence) <= 1:
        if len(sentence) == 1:
            return [[sentence]]
        else:
            return [[]]
    cut1 = []
    cut2 = []
    cut3 = []
    if sentence[:1] in Dict:
        cut1 = [ [sentence[:1]]+last_cut for last_cut in all_cut(sentence[1:], Dict)]  # 都是采用递归的方法
    if len(sentence) >= 2 and sentence[:2] in Dict:
        cut2 = [ [sentence[:2]]+last_cut for last_cut in all_cut(sentence[2:], Dict)]
    if len(sentence) >= 3 and sentence[:3] in Dict:
        cut3 = [ [sentence[:3]]+last_cut for last_cut in all_cut(sentence[3:], Dict)]
    # print("cut1分类",cut1)
    # print("cut2分类",cut2)
    return cut1 + cut2 + cut3

print(all_cut(sentence, Dict))
"""


# 第二种 结巴源码
"""
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
"""


# 第三种   队列
"""
def full_segment(text, word_dict):
    n = len(text)

    # dp[i]表示text[:i]的所有切分方式
    dp = [[] for _ in range(n + 1)]
    dp[0] = [""]

    for i in range(1, n + 1):
        for j in range(i):
            word = text[j:i]
            # print("i:",i,"j:",j,"word:",word)
            if word in word_dict:
                for prev in dp[j]:
                    # print("prev:",prev)
                    # print("---")
                    dp[i].append(prev + " " + word)

    return dp[n]
result = full_segment(sentence, Dict)
for r in result:
    print(r)
"""


# 第四种   同样采用递归
def all_cut(sentence, Dict):
    target = []
    if not sentence:
        return [[]]

    for i in range(1, len(sentence) + 1):  # 遍历从第一个字符开始到所有可能的结束位置
        word = sentence[:i]  # 提取从起始位置到当前位置i的word
        if word in Dict:  # 如果存在，则存储，并递归剩余部分
            for res in all_cut(sentence[i:], Dict):
                target.append([word] + res)

    return target


target = all_cut(sentence, Dict)
print(target)






