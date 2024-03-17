#week3作业

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
    def generate_splits_recursive(input_str, start, result, results):
        # 枚举出全部情况，保留目标输出
        if start == len(input_str) and len(''.join(result)) == len(input_str):
            # 由于回溯，需要创建数组副本
            results.append(result.copy())

        for i in range(start, len(input_str)):
            for j in range(i + 1, len(input_str) + 1):
                word = input_str[i:j]
                if word in Dict:
                    result.append(word)
                    generate_splits_recursive(input_str, j, result, results)
                    result.pop()
    results = []
    generate_splits_recursive(sentence, 0, [], results)
    return results

results = all_cut(sentence, Dict)
# 打印
for target in results:
    print(target)


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

