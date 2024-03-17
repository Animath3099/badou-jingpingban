# Dict = {"经常":0.1,
#         "经":0.05,
#         "常":0.001,
#         "有意见":0.1,
#         "有":0.1,
#         "意见":0.2,
#         "意":0.05,
#         "见":0.05,
#         "见分歧":0.05,
#         "分歧":0.2,
#         "分":0.1,
#         "歧":0.001}
# sentence = '经常有意见分歧'
def full_cut(sentence, dictionary, start=0):
    result = []

    for end in range(start, len(sentence) + 1):
        current_word = sentence[start:end]

        if current_word in dictionary:
            res = full_cut(sentence, dictionary, end)

            if res:
                for seg in res:
                    result.append([current_word] + seg)
            else:
                result.append([current_word])

    return result


Dict = {"经常": 0.1, "经": 0.05,
        "常": 0.001, "有意见": 0.1,
        "有": 0.1, "意见": 0.2,
        "意": 0.05, "见": 0.05,
        "见分歧": 0.05, "分歧": 0.2,
        "分": 0.1, "歧": 0.001}
sentence = '经常有意见分歧'

result = full_cut(sentence, Dict)
# target =[]
for cut in result:
    print(cut)
#     target.append(cut)
# print(target)