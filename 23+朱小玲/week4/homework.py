def all_cut(sentence,Dict):
    results=[]
    def dfs(current_sentence,path):
        if not current_sentence:
            results.append(path)
            return

        for i in range(1,len(current_sentence)+1): # >=1 <len(current_sentence)+1
            word = current_sentence[:i] #[:i]不包含1
            if(word in Dict):
                dfs(current_sentence[i:], path+[word])

    dfs(sentence,[])
    return results

# 测试函数
Dict = {"经常":0.1, "经":0.05, "有":0.1, "常":0.001, "有意见":0.1, "歧":0.001,
        "意见":0.2, "分歧":0.2, "见":0.05, "意":0.05, "见分歧":0.05, "分":0.1}

sentence = "经常有意见分歧"
all_cut_results = all_cut(sentence, Dict)

# 打印结果
for result in all_cut_results:
    print(result)
