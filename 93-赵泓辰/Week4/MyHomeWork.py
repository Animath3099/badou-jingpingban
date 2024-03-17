# week3作业
import math


def generate_combinations(a, n):
    # 返回在列表a中选取n个元素的所有可能组合
    def generate_combinations_helper(current_combination, remaining_elements, n):
        if n == 0:
            result.append(current_combination)
            return
        if len(remaining_elements) < n:
            return

        # 包含当前元素
        generate_combinations_helper(
            current_combination + [remaining_elements[0]],
            remaining_elements[1:],
            n - 1
        )

        # 不包含当前元素
        generate_combinations_helper(
            current_combination,
            remaining_elements[1:],
            n
        )

    result = []
    generate_combinations_helper([], a, n)
    return result


def get_result(comma_list, sentence):
    # 给定逗号的位置列表和文本，返回切分好的文本
    start = 0
    result = []
    comma_list.append(len(sentence)-1)
    for end in comma_list:
        result.append(sentence[start:end+1])
        start = end + 1
    return result


# 实现全切分函数，输出根据字典能够切分出的所有的切分方式
def all_cut(sentence, Dict):
    """
    思路：
    根据所有可能的逗号位置组合（切分位置组合），获取所有可能的切分结果
    再通过判断可能的切分结果中的词是否在词表中，来确定切分结果是否正确
    """
    max_word_len = max([len(word) for word in Dict])
    min_num_comma = math.ceil(len(sentence)/max_word_len) - 1  # 获取可能的最少逗号数量
    num_comma_list = list(range(min_num_comma, len(sentence)))  # 获取所有可能的逗号数量

    basic_comma = list(range(len(sentence)-1))  # 每个字都切分的情况下的逗号位置
    my_target = []
    for n in num_comma_list:
        comma_list = list(generate_combinations(basic_comma, n))  # 在逗号数量为n的情况下，获取所有可能的逗号位置
        for comma in comma_list:
            result = get_result(list(comma), sentence)  # 获取在给定逗号位置的情况下，文本的切分结果
            flag = True
            for word in result:
                if word not in Dict:
                    flag = False  # 判断文本切分结果是否正确
                    break
            if flag:
                my_target.append(result)  # 如果文本切分结果正确，则保存该结果
    return my_target


if __name__ == '__main__':
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

    my_target = all_cut(sentence, Dict)
    for result in my_target:
        print(result)
    print(sorted(target) == sorted(my_target))  # 判断切分结果是否正确

