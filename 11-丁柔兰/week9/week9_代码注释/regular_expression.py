import re
import random
import time

"""
介绍正则表达式的常用操作
"""

# # re.match(pattern, string) 验证字符串起始位置是否与pattern匹配
# print(re.match('www', 'wwwww.runoob.com'))         # 在起始位置匹配
# print(re.match('run', 'www.runoob.com'))         # 不在起始位置匹配

# # re.search(pattern, string) 验证字符串中是否与有片段与pattern匹配
# print(re.search('www', 'www.runoob.com'))        # 在起始位置匹配
# print(re.search('run', 'www.runoob.com'))        # 不在起始位置匹配


# #pattern中加括号，可以实现多个pattern的抽取
# line = "Cats are smarter than dogs"
# matchObj = re.match(r'(.*) are (.*?) .*', line)
# if matchObj:
#     print("matchObj.group() : ", matchObj.group())
#     print("matchObj.group(1) : ", matchObj.group(1))
#     print("matchObj.group(2) : ", matchObj.group(2))
# else:
#     print("No match!!")

###########################################

# re.sub(pattern, repl, string, count=0) 利用正则替换文本
# 将string中匹配到pattern的部分，替换为repl
# phone = "2004-959-559 # 这是一个国外电话号码"
# # 删除字符串中的 # 后注释
# num = re.sub('#.*$', "", phone)
# print("电话号码是: ", num)
# # 删除非数字(-)的字符串  \D 代表非数字  \d 代表数字
# num = re.sub('\d', "*", phone)
# print("电话号码是 : ", num)

# repl 参数可以是一个函数,要注意传入的参数不是值本身，是match对象
# 将匹配的数字乘以 2
# def double(matched):
#     return str(int(matched.group()) * 2)

# string = 'A23G4HFD567'
# print(re.sub('\d', double, string))

# count参数决定替换几次，默认是全部替换
# string = "00000"
# print(re.sub("0", "1", string, count=2))

#############################

# re.findall(string[, pos[, endpos]])
# 在字符串中找到正则表达式所匹配的所有子串，并返回一个列表，如果没有找到匹配的，则返回空列表
# pattern = re.compile('\d+')  # 查找数字
# result1 = pattern.findall('runoob 123 google 456')
# result2 = pattern.findall('run88oob123google456', 0, 10)
# print(result1)
# print(result2)

# print(re.findall("北京|上海|广东", "我从北京去上海"))

#################################

# re.split(pattern, string[, maxsplit=0]) 照能够匹配的子串将字符串分割后返回列表
# string = "1、不评价别人; 2、不给别人建议; 3、没有共同利益,不必追求共识"
# print(re.split("\d、", string))
# print(re.split(";|、", string))

###############################
# 匹配汉字  汉字unicode编码范围[\u4e00-\u9fa5]
# print(re.findall("[\u4e00-\u9fa5]", "ad噶是的12范德萨发432文"))

###############################
# 如果需要匹配，在正则表达式中有特殊含义的符号，需做转义
# print(re.search("(图)", "贾玲成中国影史票房最高女导演(图)").group())
# print(re.search("\(图\)", "贾玲成中国影史票房最高女导演(图)").group())
# print(re.sub("(图)", "", "贾玲成中国影史票房最高女导演(图)"))
# print(re.sub("\(图\)", "", "贾玲成中国影史票房最高女导演(图)"))

################################
# pattern = "\d12\w"
# re_pattern = re.compile(pattern)
# print(re.search(pattern, "432312d"))


# 效率
'''
这段代码的目的是为了给出在长字符串中查找短字符串时，使用预编译的正则表达式与使用Python内置的in关键字比较的性能测试结果。通过对比两者的耗时，可以了解在特定情况下哪种方法更加高效
'''
# 导入time模块用于计时，random模块用于生成随机数据
import time
import random

# 创建一个包含英文字母表的列表
chars = list("abcdefghijklmnopqrstuvwxyz")
# 随机生成长度为n的字母组成的字符串
# 使用列表推导式随机选择100个字母，并将它们连接成一个长字符串
string = "".join([random.choice(chars) for i in range(100)])
# 同样，随机选择4个字母，并将它们连接成一个较短的字符串，这将作为待查找的模式
pattern = "".join([random.choice(chars) for i in range(4)])
# 使用正则表达式的compile方法预编译待查找的模式，以提高查找效率
re_pattern = re.compile(pattern)
start_time = time.time()  # 记录开始查找的时间
for i in range(50000):  # 设置一个循环，进行50000次查找操作
    # 在循环中使用正则表达式模块的search方法查找预编译的模式在长字符串中的匹配。
    # 被注释的两行代码是之前的测试代码，用于每次循环都生成一个新的模式并进行查找，但当前测试中不需要这样做
    # pattern = "".join([random.choice(chars) for i in range(3)])
    # re.search(pattern, string)
    re.search(re_pattern, string)
# 打印正则表达式查找所花费的总时间
print("正则查找耗时：", time.time() - start_time)
# 重置开始查找的时间，以便于测试下一种查找方法
start_time = time.time()
for i in range(50000):  # 再次设置一个循环，进行50000次查找操作
    # 在循环中使用Python的in关键字来检查短字符串是否存在于长字符串中。同样，被注释的代码是之前的测试代码，在当前测试中不使用
    # pattern = "".join([random.choice(chars) for i in range(3)])
    pattern in string
# 使用Python in关键字查找所花费的总时间
print("python in查找耗时：", time.time() - start_time)
