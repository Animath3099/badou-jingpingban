import torch
import math
import numpy as np
from transformers import BertModel

'''

通过手动矩阵运算实现Bert结构
模型文件下载 https://huggingface.co/models

'''
# 从人家训练好的模型里把权重拿出来
bert = BertModel.from_pretrained(r"D:\BaiduNetdiskDownload\八斗课程-精品班\第六周\week6 语言模型和预训练\下午\bert-base-chinese", return_dict=False)
state_dict = bert.state_dict()
bert.eval()
x = np.array([2450, 15486, 102, 2110]) #通过vocab对应输入：深度学习
torch_x = torch.LongTensor([x])  #pytorch形式输入 1*4
seqence_output, pooler_output = bert(torch_x)  # 1*4*768（1个样本，样本长度维4，每个字的向量维度768，4个字）  1*768
# print(seqence_output.shape, pooler_output.shape)
# print(seqence_output, pooler_output)

# print(bert.state_dict().keys())  #查看所有的权值矩阵名称


#softmax归一化
def softmax(x):
    return np.exp(x)/np.sum(np.exp(x), axis=-1, keepdims=True)

#gelu激活函数
def gelu(x):
    return 0.5 * x * (1 + np.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * np.power(x, 3))))

class DiyBert:
    #将预训练好的整个权重字典输入进来
    def __init__(self, state_dict):
        self.num_attention_heads = 12
        self.hidden_size = 768
        self.num_layers = 12
        self.load_weights(state_dict)

    def load_weights(self, state_dict):
        #embedding部分
        self.word_embeddings = state_dict["embeddings.word_embeddings.weight"].numpy()  # 21128,768
        # vocab_size * hidden_size 词表字表的embedding

        self.position_embeddings = state_dict["embeddings.position_embeddings.weight"].numpy()  # 512,768
        # max_position_embeddings * hidden_size 顺序的embedding

        self.token_type_embeddings = state_dict["embeddings.token_type_embeddings.weight"].numpy()  # 2,768
        # type_vocab_size * hidden_size 位置的embedding

        # embedding层的归一化层
        self.embeddings_layer_norm_weight = state_dict["embeddings.LayerNorm.weight"].numpy()  # 768,
        self.embeddings_layer_norm_bias = state_dict["embeddings.LayerNorm.bias"].numpy()  # 768,
        self.transformer_weights = []
        #transformer部分，有多层，这里进行参数的初始化，有12次
        for i in range(self.num_layers):  # 12层，与后面的对应
            q_w = state_dict["encoder.layer.%d.attention.self.query.weight" % i].numpy() # 768*768
            # hidden_size * hidden_size
            # print(q_w.shape)
            q_b = state_dict["encoder.layer.%d.attention.self.query.bias" % i].numpy() # 768,
            k_w = state_dict["encoder.layer.%d.attention.self.key.weight" % i].numpy()
            k_b = state_dict["encoder.layer.%d.attention.self.key.bias" % i].numpy()
            v_w = state_dict["encoder.layer.%d.attention.self.value.weight" % i].numpy()
            v_b = state_dict["encoder.layer.%d.attention.self.value.bias" % i].numpy()
            attention_output_weight = state_dict["encoder.layer.%d.attention.output.dense.weight" % i].numpy()
            attention_output_bias = state_dict["encoder.layer.%d.attention.output.dense.bias" % i].numpy()
            attention_layer_norm_w = state_dict["encoder.layer.%d.attention.output.LayerNorm.weight" % i].numpy()
            attention_layer_norm_b = state_dict["encoder.layer.%d.attention.output.LayerNorm.bias" % i].numpy()
            intermediate_weight = state_dict["encoder.layer.%d.intermediate.dense.weight" % i].numpy()
            intermediate_bias = state_dict["encoder.layer.%d.intermediate.dense.bias" % i].numpy()
            output_weight = state_dict["encoder.layer.%d.output.dense.weight" % i].numpy()
            output_bias = state_dict["encoder.layer.%d.output.dense.bias" % i].numpy()
            ff_layer_norm_w = state_dict["encoder.layer.%d.output.LayerNorm.weight" % i].numpy()
            ff_layer_norm_b = state_dict["encoder.layer.%d.output.LayerNorm.bias" % i].numpy()

            # 参数弄出来之后放到这个列表里，之后transformer的时候提取出来用，有12批
            self.transformer_weights.append([q_w, q_b, k_w, k_b, v_w, v_b, attention_output_weight, attention_output_bias,
                                             attention_layer_norm_w, attention_layer_norm_b, intermediate_weight, intermediate_bias,
                                             output_weight, output_bias, ff_layer_norm_w, ff_layer_norm_b])
        #pooler层
        self.pooler_dense_weight = state_dict["pooler.dense.weight"].numpy()
        self.pooler_dense_bias = state_dict["pooler.dense.bias"].numpy()


    #bert embedding，使用3层叠加，在经过一个embedding层
    def embedding_forward(self, x):
        # x.shape = [max_len]
        # print(len(x))
        we = self.get_embedding(self.word_embeddings, x)  # shpae: [max_len, hidden_size]
        # 根据x在vocab中选择max_len这么多的向量，len(x) * hidden_size

        # position embeding的输入 [0, 1, 2, 3]
        pe = self.get_embedding(self.position_embeddings, np.array(list(range(len(x)))))  # shpae: [max_len, hidden_size]
        # token type embedding,单输入的情况下为[0, 0, 0, 0]
        te = self.get_embedding(self.token_type_embeddings, np.array([0] * len(x)))  # shpae: [max_len, hidden_size]
        embedding = we + pe + te
        # 加和后有一个归一化层
        embedding = self.layer_norm(embedding, self.embeddings_layer_norm_weight, self.embeddings_layer_norm_bias)  # shpae: [max_len, hidden_size]
        return embedding

    #embedding层实际上相当于按index索引，或理解为onehot输入乘以embedding矩阵
    def get_embedding(self, embedding_matrix, x):
        return np.array([embedding_matrix[index] for index in x])
    #在茫茫词表中，选取x中的四个位置的字词，形成max_len, hidden_size形状



    #执行全部的transformer层计算
    def all_transformer_layer_forward(self, x):
        for i in range(self.num_layers):  # num_layers是12，所以执行12层的计算
            x = self.single_transformer_layer_forward(x, i)  # 单层，但是循环12次
        return x


    #执行单层transformer层计算，因为transformer层有很多块，都写在这里了，详细的还有自己的函数
    def single_transformer_layer_forward(self, x, layer_index):
        weights = self.transformer_weights[layer_index]  # 通过layer_index索引得到self.transformer_weights中存储的所有参数。
        #取出该层的参数，在实际中，这些参数都是随机初始化，之后进行预训练
        q_w, q_b, \
        k_w, k_b, \
        v_w, v_b, \
        attention_output_weight, attention_output_bias, \
        attention_layer_norm_w, attention_layer_norm_b, \
        intermediate_weight, intermediate_bias, \
        output_weight, output_bias, \
        ff_layer_norm_w, ff_layer_norm_b = weights  # 逐一提取参数并赋值给前面这堆
        #self attention层
        attention_output = self.self_attention(x,
                                q_w, q_b,
                                k_w, k_b,
                                v_w, v_b,
                                attention_output_weight, attention_output_bias,
                                self.num_attention_heads,
                                self.hidden_size)
        #bn层，并使用了残差机制（函数是归一化层），其中输入了X+Z，也就是原始数据与自注意力输出的数据
        #是self attention之后的layernorm层
        x = self.layer_norm(x + attention_output, attention_layer_norm_w, attention_layer_norm_b)  # 4*768
        #feed forward层
        feed_forward_x = self.feed_forward(x,
                              intermediate_weight, intermediate_bias,
                              output_weight, output_bias)

        #bn层，并使用了残差机制 同样进行了一次上一步数据与新数据的相加
        # max_len, hidden_size 经过两个参数，参数形状是hidden_size, 但是会进行广播，将w赋值为4*768的形状，
        # 后与归一化的x进行元素级别的相乘，最终得到还是 max_len, hidden_size 的
        x = self.layer_norm(x + feed_forward_x, ff_layer_norm_w, ff_layer_norm_b)
        return x

    # self attention的计算
    def self_attention(self,
                       x,
                       q_w,
                       q_b,
                       k_w,
                       k_b,
                       v_w,
                       v_b,
                       attention_output_weight,
                       attention_output_bias,
                       num_attention_heads,
                       hidden_size):
        # x.shape = max_len * hidden_size
        # q_w, k_w, v_w  shape = hidden_size * hidden_size
        # q_b, k_b, v_b  shape = hidden_size
        q = np.dot(x, q_w.T) + q_b  # shape: [max_len, hidden_size]      W * X + B lINER
        k = np.dot(x, k_w.T) + k_b  # shpae: [max_len, hidden_size]
        v = np.dot(x, v_w.T) + v_b  # shpae: [max_len, hidden_size]
        attention_head_size = int(hidden_size / num_attention_heads)
        # q.shape = num_attention_heads, max_len, attention_head_size  12 * 4 * 64  分成12份
        q = self.transpose_for_scores(q, attention_head_size, num_attention_heads)
        # k.shape = num_attention_heads, max_len, attention_head_size
        k = self.transpose_for_scores(k, attention_head_size, num_attention_heads)
        # v.shape = num_attention_heads, max_len, attention_head_size
        v = self.transpose_for_scores(v, attention_head_size, num_attention_heads)
        # qk.shape = num_attention_heads, max_len, max_len
        qk = np.matmul(q, k.swapaxes(1, 2))  # 12*4*64 与 12*64*4 得到 12*4*4
        qk /= np.sqrt(attention_head_size)
        qk = softmax(qk)  # 12*4*4
        # qkv.shape = num_attention_heads, max_len, attention_head_size
        qkv = np.matmul(qk, v)  # 12*4*64
        # qkv.shape = max_len, hidden_size
        qkv = qkv.swapaxes(0, 1).reshape(-1, hidden_size)  # 拼接十二个

        # attention.shape = max_len, hidden_size
        # 多头拼接之后，进入这个线性层 output = linear(attention(Q,K,V))
        attention = np.dot(qkv, attention_output_weight.T) + attention_output_bias
        # 4*768与768*768，加上扩散成1*768的bias，变成4*768
        return attention

    #多头机制
    def transpose_for_scores(self, x, attention_head_size, num_attention_heads):
        # hidden_size = 768  num_attent_heads = 12 attention_head_size = 64
        max_len, hidden_size = x.shape  # 右边 max_len, hidden_size
        x = x.reshape(max_len, num_attention_heads, attention_head_size)
        x = x.swapaxes(1, 0)  # output shape = [num_attention_heads, max_len, attention_head_size]
        return x

    #前馈网络的计算
    def feed_forward(self,
                     x,
                     intermediate_weight,  # intermediate_size, hidden_size
                     intermediate_bias,  # intermediate_size
                     output_weight,  # hidden_size, intermediate_size
                     output_bias,  # hidden_size
                     ):
        # output shpae: [max_len, intermediate_size]
        # [max_len, hidden_size] 乘 [hidden_size, intermediate_size] 再加上 [intermediate_size]
        x = np.dot(x, intermediate_weight.T) + intermediate_bias  # 4*768乘上768*3072得到4*3072
        x = gelu(x)  # max_len, intermediate_size 经过激活，激活函数是gelu

        # output shpae: [max_len, hidden_size]
        # [max_len, intermediate_size] * [intermediate_size, hidden_size]
        # 其中 output_weight本来是hidden_size, intermediate_size，转置了
        x = np.dot(x, output_weight.T) + output_bias
        return x

    #归一化层
    # x是4*768进来的，max_len, hidden_size w和b进行了广播操作之后进行了元素级别的运算
    def layer_norm(self, x, w, b):
        x = (x - np.mean(x, axis=1, keepdims=True)) / np.std(x, axis=1, keepdims=True)
        x = x * w + b
        return x

    #链接[cls] token的输出层
    def pooler_output_layer(self, x):
        # 1, hidden_size 和 hidden_size，hidden_size
        x = np.dot(x, self.pooler_dense_weight.T) + self.pooler_dense_bias
        x = np.tanh(x)
        return x
    # 输出的是一个字

    #最终输出
    def forward(self, x):
        x = self.embedding_forward(x)  # embedding层，三个embedding相加
        sequence_output = self.all_transformer_layer_forward(x)  # 十二层transformer，输出max_len, hidden_size
        print(sequence_output)
        pooler_output = self.pooler_output_layer(sequence_output[0])  # 输入了第一个字
        print(pooler_output)
        return sequence_output, pooler_output


#自制
db = DiyBert(state_dict)
diy_sequence_output, diy_pooler_output = db.forward(x)
#torch
torch_sequence_output, torch_pooler_output = bert(torch_x)

# print("diy_sequence_output", diy_sequence_output)
# print(torch_sequence_output)

# print(diy_pooler_output)
# print(torch_pooler_output)