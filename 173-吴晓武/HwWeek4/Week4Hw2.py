import  re
import time


def all_cut(sentence, Dict):
    def dfs(sentence, path, result):
        if not sentence:
            result.append(path)
            return
        for i in range(1 , len(sentence)+1):
            word =  sentence[:i]

            if word in Dict:
                dfs(sentence[i:],path+[word],result)

    result = []
    dfs(sentence,[],result)
    return  result

sentence = "经常有意见分歧"

Dict = {"经常",
        "经",
        "有",
        "常",
        "有意见",
        "歧",
        "意见",
        "分歧",
        "见",
        "意",
        "见分歧",
        "分"}

target = all_cut(sentence, Dict)
for t in target:
      print(t)


