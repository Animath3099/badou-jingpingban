#Week4 HomeWork

dict = {'经常':1,
    '经':0,
    '常':0,
    '有':0,
    '意':0,
    '见':0,
    '分':0,
    '歧':0,
    '有意见':1,
    '意见':1,
    '分歧':1,
    '见分歧':1}


sentence = '经常有意见分歧'
max_length = len(max(dict, key = len))

#正向
def partition(sentence, dictionary):
    result = []
    max_length = len(max(dictionary, key=len))
    backtracking(sentence, 0, [], result, dictionary, max_length)
    return result

def backtracking(s, start_index, path, result, dictionary, max_length):
    if start_index == len(s):
        result.append(path)
        return

    for i in range(start_index, min(start_index + max_length, len(s))):
        prefix = s[start_index:i + 1]
        if prefix in dictionary:
            backtracking(s, i + 1, path + [prefix], result, dictionary, max_length)

#反向
def backtracing_reverse(sentence, path, result, endindex, max_length):
    if endindex == -1:
        result.append(path)  # 倒序添加到结果中
        return
    
    for i in range(endindex, max(endindex - max_length, -1), -1):  # 从末尾向前遍历
        temp = sentence[i:endindex+1]
        if temp in dict:
            backtracing_reverse(sentence, path+[temp], result, i - 1, max_length)

def function_reverse(sentence, max_length):
    result = []
    backtracing_reverse(sentence,[],result, len(sentence) - 1, max_length)
    return result