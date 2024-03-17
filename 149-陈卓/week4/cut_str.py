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

sentence = "经常有意见分歧"

def all_cut(sentence, Dict, start=0, path=None, result=None):
    if result is None:
        result = []
    if path is None:
        path = []
    
    if start == len(sentence):
        result.append(path)
        return result
    
    for end in range(start + 1, len(sentence) + 1):
        word = sentence[start:end]
        if word in Dict:
            all_cut(sentence, Dict, end, path + [word], result)

    return result

cut = all_cut(sentence, Dict)
print(cut)