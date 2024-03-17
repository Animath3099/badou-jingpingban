# week3作业
import numpy as np
import jieba
import paddle
# paddle.enable_static()
# jieba.enable_paddle()# 启动paddle模式。 0.40版之后开始支持，早期版本不支持
strs=["经常有意见分歧","我来到北京清华大学","乒乓球拍卖完了","中国科学技术大学"]
for str in strs:
    seg_list = jieba.cut(str,use_paddle=True) # 使用paddle模式
    print("Paddle Mode: " + '/'.join(list(seg_list)))

seg_list = jieba.cut("我来到北京清华大学", cut_all=True)
print("Full Mode: " + "/ ".join(seg_list))  # 全模式

seg_list = jieba.cut("我来到北京清华大学", cut_all=False)
print("Default Mode: " + "/ ".join(seg_list))  # 精确模式

seg_list = jieba.cut("他来到了网易杭研大厦")  # 默认是精确模式
print(", ".join(seg_list))

seg_list = jieba.cut_for_search("小明硕士毕业于中国科学院计算所，后在日本京都大学深造")  # 搜索引擎模式
print(", ".join(seg_list))
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
# sentence = "经常有意见分歧"


sentence = "经常有意见分歧"



# 实现全切分函数，输出根据字典能够切分出的所有的切分方式
def all_cut(sentence,dict):

    target=[]

    _all_cut(sentence,[],0,target,dict)

    for k in target:
        print(k)



def _all_cut(items,list,i,target,dict):

    #i为游标  i走到头 输出list到target
    if i == len(items):
        # print('print list:',list)
        target.append(list.copy())
        return


    for j in range(i,len(items)):
        # print(items[i:j+1],',',i,',')

        #取出一个切片
        tmp = items[i:j+1]
        #判断切片是否在词表里，如果不在  则跳过并继续
        if tmp not in dict.keys():
            continue

        #如果在则加入list中存储
        list.append(tmp)



        #取剩下切片的全切片
        _all_cut(items,list,j+1,target,dict)

        # print(i,'--',list)
        # 取出list中多余的元素 重新构造切片
        # print('pop:',list.pop())
        list.pop()

# 输入字符串和字典，返回词的列表


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

if __name__ == '__main__':
    all_cut(sentence,Dict)

