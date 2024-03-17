#week3作业

#词典；每个词后方存储的是其词频，词频仅为示例，不会用到，也可自行修改
dict = {"经常":0.1,
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
def all_cut(sentence, dict):

    n = len(sentence)
    cut = [False] * (n+1)
    cut_res = [[] for _ in range(n + 1)]
    cut[0] = True
    for j in range(1, n+1):
        # if results != []:

        for i in range(j):
            # words = []
            word = sentence[i:j]
            # print('哈哈哈哈', word)
            if word in dict and cut[i]:
                cut[j] = True
                if i == 0:
                    cut_res[j].append(word)
                else:
                    for item in cut_res[i]:
                        cut_res[j].append(item + '/' + word)

        # print(cut)
        # count += 1
        # print(count)
        # words.append(word)
        # sentence = sentence[len(word):]
    return cut_res[n]

result = all_cut(sentence, dict)
for i in range(len(result)):
    print(result[i])

# 目标输出;顺序不重要
# target = [
#     ['经常', '有意见', '分歧'],
#     ['经常', '有意见', '分', '歧'],
#     ['经常', '有', '意见', '分歧'],
#     ['经常', '有', '意见', '分', '歧'],
#     ['经常', '有', '意', '见分歧'],
#     ['经常', '有', '意', '见', '分歧'],
#     ['经常', '有', '意', '见', '分', '歧'],
#     ['经', '常', '有意见', '分歧'],
#     ['经', '常', '有意见', '分', '歧'],
#     ['经', '常', '有', '意见', '分歧'],
#     ['经', '常', '有', '意见', '分', '歧'],
#     ['经', '常', '有', '意', '见分歧'],
#     ['经', '常', '有', '意', '见', '分歧'],
#     ['经', '常', '有', '意', '见', '分', '歧']
# ]