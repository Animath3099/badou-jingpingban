# week4作业
import copy

# 词典；每个词后方存储的是其词频，词频仅为示例，不会用到，也可自行修改
Dict = {"经常": 0.1,
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

# 待切分文本
sentence = "经常有意见分歧"

# 目标输出;顺序不重要
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


# 实现全切分函数，输出根据字典能够切分出的所有的切分方式

def dfs(result, current_list, string, word_dict, left, right):
    copy_list = copy.deepcopy(current_list)
    print_list(copy_list)
    if "".join(copy_list) == string:
        print('add to result')
        result.append(copy_list)
        return

    while right - left <= max_word_length and right <= len(string):
        dfs(result, copy_list, string, word_dict, left, right + 1)
        if string[left:right] in word_dict.keys():
            copy_list.append(string[left:right])
            dfs(result, copy_list, string, word_dict, right, right + 1)
            break
        else:
            break


def max_len(Dict):
    max_l = 0
    for word in Dict.keys():
        max_l = max(max_l, len(word))
    print("max word length: ", max_l)
    return max_l


max_word_length = max_len(Dict)

def all_cut(sentence, Dict):
    result = []
    dfs(result, [], sentence, Dict, 0, 1)
    return result


def print_list(l):
    print(" / ".join(l))


if __name__ == "__main__":
    output = all_cut(sentence, Dict)
    print("Total combinations: ", len(output))
    print("Target size: ", len(target))
    print("All combinations: ")
    for comb in output:
        print_list(comb)
