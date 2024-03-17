#coding:utf8
import torch
import torch.nn as nn

def to_sequence(strs,voc_list_auto):
    sequence = []
    for str in strs:
        sequence.append([voc_list_auto[char] for char in str])
    return sequence

def padding_seq(sequence_num):
    max_length = max(len(sequence) for sequence in sequence_num)
    res = []
    for sequence in sequence_num:
        if len(sequence) >= max_length:
            sequence = sequence[:max_length]
        else:
            sequence = sequence + [0] * (max_length - len(sequence))
        res.append(sequence)
    return res

def print_res(data_label,data):
    print(f"{data_label}:\n{data}")
    print('****************************************')
str = 'abcdefghijklmnopqrstuvwxyz'
voc_list_auto = {
        '[pad]': 0,
}
index = 1
for char in str:
    voc_list_auto[char] = index
    index += 1
#print(voc_list_auto)
str_list = ['agcd', 'abcdefghijklmnopq', 'jkjkijda']

sequence_num = to_sequence(str_list, voc_list_auto)
sequence_num = padding_seq(sequence_num)
sequence_num = torch.tensor(sequence_num)
#print(sequence_num)
num_embeddings, embedding_dims = 27, 10
'''embedding层相关参数如下：
        num_embeddings:需要的生成向量的个数
        embedding_dims:生成的向量的维度数'''
embedding_layer = nn.Embedding(num_embeddings, embedding_dims)
print_res('embedding_layer.weight' ,embedding_layer.weight)
#sequence_num = torch.flatten(sequence_num)
embedding_input = torch.LongTensor(sequence_num)
print_res('embedding_input',embedding_input)
embedding_output = embedding_layer(embedding_input)
print_res('embedding_output',embedding_output)

