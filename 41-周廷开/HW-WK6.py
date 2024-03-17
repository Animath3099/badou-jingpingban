import torch
import math
import numpy as np
from transformers import BertModel
import torch.nn as nn
'''
计算bert参数量，直接运行可在命令行查看结果，总参数量为24301056
'''

bert = BertModel.from_pretrained(r"D:\八斗ai\第六周\week6 语言模型和预训练\week6 语言模型和预训练\HW\bert-base-chinese\bert-base-chinese", return_dict=False)
state_dict = bert.state_dict()
bert.eval()
x = np.array([2450, 15486, 102, 2110]) #通过vocab对应输入：深度学习
torch_x = torch.LongTensor([x])  #pytorch形式输入
# seqence_output, pooler_output = bert(torch_x)
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
        self.num_layers = 1
        self.load_weights(state_dict)

    def load_weights(self, state_dict):

        #embedding部分
        self.word_embeddings = state_dict["embeddings.word_embeddings.weight"].numpy()
        #word_embeddings参数量计算 - begin
        vocab_size_word, hidden_size_word = self.word_embeddings.shape
        word_embeddings_para = vocab_size_word * hidden_size_word
        print("1.word_embeddings层的形状为{}(vocab_size_word) * {}(hidden_size_word)".format
              (vocab_size_word,hidden_size_word))
        print("  前者由预训练所使用的词表决定，后者为训练设置的参数")
        print("  因此word_embeddings层的参数量为{}*{}={}".format
              (vocab_size_word,hidden_size_word,word_embeddings_para))
        #word_embeddings参数量计算 - end

        self.position_embeddings = state_dict["embeddings.position_embeddings.weight"].numpy()
        # position_embeddings参数量计算 - begin
        vocab_size_position, hidden_size_position = self.position_embeddings.shape
        position_embeddings_para = vocab_size_position * hidden_size_position
        print("2.position_embeddings层的形状为{}(max_position_embeddings) * {}(hidden_size)".format
              (vocab_size_position, hidden_size_position))
        print("  前者由预训练config中的max_position_embeddings参数决定，"
              "代表模型能处理的最大序列长度，后者仍然为训练设置的hidden_size")
        print("  因此position_embeddings层的参数量为{}*{}={}".format
              (vocab_size_position, hidden_size_position, position_embeddings_para))
        # position_embeddings参数量计算 - end

        self.token_type_embeddings = state_dict["embeddings.token_type_embeddings.weight"].numpy()
        # token_type_embeddings参数量计算 - begin
        vocab_size_token_type, hidden_size_token_type = self.token_type_embeddings.shape
        token_type_embeddings_para = vocab_size_token_type * hidden_size_token_type
        print("3.position_embeddings层的形状为{}(type_vocab_size) * {}(hidden_size)".format
              (vocab_size_token_type, hidden_size_token_type))
        print("  前者由预训练config中的type_vocab_size参数决定，"
              "代表模型能处理的最大序列长度，后者仍然为训练设置的hidden_size")
        print("  因此token_type_embeddings层的参数量为{}*{}={}".format
              (vocab_size_token_type, hidden_size_token_type, token_type_embeddings_para))
        # token_type_embeddings参数量计算 - end

        self.embeddings_layer_norm_weight = state_dict["embeddings.LayerNorm.weight"].numpy()
        #embeddings_layer_norm_weight参数量计算 - begin
        ln_size = self.embeddings_layer_norm_weight.shape[0]
        ln_embeddings_para = ln_size
        print("4.embeddings_layer_norm_weight层的形状为{}(hidden_size) * 1".format
              (ln_size))
        print("  LN层对张量进行降维后，形状变为hidden_size*1")
        print("  因此embeddings_layer_norm_weight层的参数量为{}*1={}".format
              (ln_size, ln_embeddings_para))
        #embeddings_layer_norm_weight参数量计算 - end

        self.embeddings_layer_norm_bias = state_dict["embeddings.LayerNorm.bias"].numpy()
        # embeddings_layer_norm_bias参数量计算 - begin
        ln_bias_size = self.embeddings_layer_norm_weight.shape[0]
        ln_bias_embeddings_para = ln_bias_size
        print("5.embeddings_layer_norm_bias层的形状为{}(hidden_size) * 1".format
              (ln_bias_size))
        print("  LN层对张量进行降维后，形状变为hidden_size*1,bias与weight一致")
        print("  因此embeddings_layer_norm_bias层的参数量为{}*1={}".format
              (ln_bias_size, ln_bias_embeddings_para))
        # embeddings_layer_norm_bias参数量计算 - end

        sum1 = (word_embeddings_para + position_embeddings_para + token_type_embeddings_para
                        + ln_embeddings_para + ln_bias_embeddings_para)
        print("sum1:embedding层总参数量为:{}+{}+{}+{}+{}={}".format(word_embeddings_para, position_embeddings_para,
            token_type_embeddings_para,ln_embeddings_para,ln_bias_embeddings_para,sum1))


        self.transformer_weights = []
        #transformer部分，有多层
        sum2 = 0
        sum3 = 0
        for i in range(self.num_layers):
            q_w = state_dict["encoder.layer.%d.attention.self.query.weight" % i].numpy()
            # q_w参数量计算 - begin
            q_w_hidden_size, _ = q_w.shape
            q_w_para = q_w_hidden_size * q_w_hidden_size
            print("6.q_w层的形状为{}(hidden_size) * {}(hidden_size)".format
                  (q_w_hidden_size, q_w_hidden_size))
            print("  因此q_w层的参数量为{}*{}={}".format
                  (q_w_hidden_size, q_w_hidden_size, q_w_para))
            print("  b的形状为768*1，并且q、k、v一样，后面不再重复计算")
            # q_w参数量计算 - end

            q_b = state_dict["encoder.layer.%d.attention.self.query.bias" % i].numpy()
            k_w = state_dict["encoder.layer.%d.attention.self.key.weight" % i].numpy()
            k_b = state_dict["encoder.layer.%d.attention.self.key.bias" % i].numpy()
            v_w = state_dict["encoder.layer.%d.attention.self.value.weight" % i].numpy()
            v_b = state_dict["encoder.layer.%d.attention.self.value.bias" % i].numpy()

            sum2 = q_w_para * 3 + 768 * 3
            print("sum2:qkv总参数量为:{}*6={}".format(q_w_para, sum2))

            attention_output_weight = state_dict["encoder.layer.%d.attention.output.dense.weight" % i].numpy()
            # attention_output参数量计算 - begin
            attention_output_weight_size1, attention_output_weight_size2 = attention_output_weight.shape
            attention_output_weight_para = attention_output_weight_size1 * attention_output_weight_size2
            print("7.attention_output_weight层的形状为{}(hidden_size) * {}(hidden_size)".format
                  (attention_output_weight_size1, attention_output_weight_size2))
            print("  因此attention_output_weight层的参数量为{}*{}={}".format
                  (attention_output_weight_size1, attention_output_weight_size2, attention_output_weight_para))

            attention_output_bias = state_dict["encoder.layer.%d.attention.output.dense.bias" % i].numpy()
            attention_output_bias_para = 768
            print("8.attention_output_bias参数量为{}".format(attention_output_bias_para))
            # attention_output参数量计算 - end

            attention_layer_norm_w = state_dict["encoder.layer.%d.attention.output.LayerNorm.weight" % i].numpy()
            attention_layer_norm_w_para = ln_embeddings_para
            print("9.attention_layer_norm_w参数量与4同理，为{}".format(attention_layer_norm_w_para))

            attention_layer_norm_b = state_dict["encoder.layer.%d.attention.output.LayerNorm.bias" % i].numpy()
            attention_layer_norm_b_para = ln_bias_size
            print("10.attention_layer_norm_b参数量与5同理，为{}".format(attention_layer_norm_b_para))

            intermediate_weight = state_dict["encoder.layer.%d.intermediate.dense.weight" % i].numpy()
            intermediate_weight_size1, intermediate_weight_size2 = intermediate_weight.shape
            print("11.intermediate_weight层的形状为{} * {}".format
                  (intermediate_weight_size1, intermediate_weight_size2))
            intermediate_weight_para = intermediate_weight_size1 * intermediate_weight_size2
            print("增加一个线性层映射到高维，增加可学习的参数量")
            print("  因此intermediate_weight层的参数量为{}*{}={}".format
                  (intermediate_weight_size1, intermediate_weight_size2, intermediate_weight_para))

            intermediate_bias = state_dict["encoder.layer.%d.intermediate.dense.bias" % i].numpy()
            intermediate_bias_para = 3072
            print("12.intermediate_bias参数量为{}".format(intermediate_bias_para))

            output_weight = state_dict["encoder.layer.%d.output.dense.weight" % i].numpy()
            output_weight_size1, output_weight_size2 = output_weight.shape
            print("13.output_weight层的形状为{} * {}".format
                  (output_weight_size1, output_weight_size2))
            output_weight_para = output_weight_size1 * output_weight_size2
            print("这个线性层是在映射到高维后，再映射回原来的形状")
            print("  因此intermediate_weight层的参数量为{}*{}={}".format
                  (output_weight_size1, output_weight_size2, output_weight_para))

            output_bias = state_dict["encoder.layer.%d.output.dense.bias" % i].numpy()
            output_bias_para = 768
            print("14.intermediate_bias参数量为{}".format(output_bias_para))

            ff_layer_norm_w = state_dict["encoder.layer.%d.output.LayerNorm.weight" % i].numpy()
            ff_layer_norm_b = state_dict["encoder.layer.%d.output.LayerNorm.bias" % i].numpy()
            ff_ln_shape = ff_layer_norm_w.shape[0]
            print("15.最后线性层ln的参数量与前面ln层一致，w和b的总参数量为{}+{}={}".format(ff_ln_shape, ff_ln_shape, ff_ln_shape*2))
            self.transformer_weights.append([q_w, q_b, k_w, k_b, v_w, v_b, attention_output_weight, attention_output_bias,
                                             attention_layer_norm_w, attention_layer_norm_b, intermediate_weight, intermediate_bias,
                                             output_weight, output_bias, ff_layer_norm_w, ff_layer_norm_b])

            sum3 = (attention_output_weight_para + attention_output_bias_para + attention_layer_norm_w_para + attention_layer_norm_b_para
                    + intermediate_weight_para + intermediate_bias_para + output_weight_para + output_bias_para + ff_ln_shape*2)
            print("sum3:transformer层的参数量为：{}".format(sum3))
        #pooler层
        self.pooler_dense_weight = state_dict["pooler.dense.weight"].numpy()
        print(self.pooler_dense_weight.shape)
        self.pooler_dense_bias = state_dict["pooler.dense.bias"].numpy()
        print("16.最后pooler层weight的参数量为768 * 768，bias参数量为768")
        sum4 = 768 * 768 + 768
        print("sum4:pooler层的参数量为：{}".format(sum4))
        sum_all = sum1 + sum2 +sum3 + sum4
        print("手动计算总参数量为：{}(sum1)+{}(sum2)+{}(sum3)+{}(sum4)={}(sum_all)".format(sum1, sum2, sum3, sum4, sum_all))

    #bert embedding，使用3层叠加，在经过一个embedding层
    def embedding_forward(self, x):
        # x.shape = [max_len]
        we = self.get_embedding(self.word_embeddings, x)  # shpae: [max_len, hidden_size]
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

    #执行全部的transformer层计算
    def all_transformer_layer_forward(self, x):
        for i in range(self.num_layers):
            x = self.single_transformer_layer_forward(x, i)
        return x

    #执行单层transformer层计算
    def single_transformer_layer_forward(self, x, layer_index):
        weights = self.transformer_weights[layer_index]
        #取出该层的参数，在实际中，这些参数都是随机初始化，之后进行预训练
        q_w, q_b, \
        k_w, k_b, \
        v_w, v_b, \
        attention_output_weight, attention_output_bias, \
        attention_layer_norm_w, attention_layer_norm_b, \
        intermediate_weight, intermediate_bias, \
        output_weight, output_bias, \
        ff_layer_norm_w, ff_layer_norm_b = weights
        #self attention层
        attention_output = self.self_attention(x,
                                q_w, q_b,
                                k_w, k_b,
                                v_w, v_b,
                                attention_output_weight, attention_output_bias,
                                self.num_attention_heads,
                                self.hidden_size)
        #bn层，并使用了残差机制
        x = self.layer_norm(x + attention_output, attention_layer_norm_w, attention_layer_norm_b)
        #feed forward层
        feed_forward_x = self.feed_forward(x,
                              intermediate_weight, intermediate_bias,
                              output_weight, output_bias)
        #bn层，并使用了残差机制
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
        # q.shape = num_attention_heads, max_len, attention_head_size
        q = self.transpose_for_scores(q, attention_head_size, num_attention_heads)
        # k.shape = num_attention_heads, max_len, attention_head_size
        k = self.transpose_for_scores(k, attention_head_size, num_attention_heads)
        # v.shape = num_attention_heads, max_len, attention_head_size
        v = self.transpose_for_scores(v, attention_head_size, num_attention_heads)
        # qk.shape = num_attention_heads, max_len, max_len
        qk = np.matmul(q, k.swapaxes(1, 2))
        qk /= np.sqrt(attention_head_size)
        qk = softmax(qk)
        # qkv.shape = num_attention_heads, max_len, attention_head_size
        qkv = np.matmul(qk, v)
        # qkv.shape = max_len, hidden_size
        qkv = qkv.swapaxes(0, 1).reshape(-1, hidden_size)
        # attention.shape = max_len, hidden_size
        attention = np.dot(qkv, attention_output_weight.T) + attention_output_bias
        return attention

    #多头机制
    def transpose_for_scores(self, x, attention_head_size, num_attention_heads):
        # hidden_size = 768  num_attent_heads = 12 attention_head_size = 64
        max_len, hidden_size = x.shape
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
        x = np.dot(x, intermediate_weight.T) + intermediate_bias
        x = gelu(x)
        # output shpae: [max_len, hidden_size]
        x = np.dot(x, output_weight.T) + output_bias
        return x

    #归一化层
    def layer_norm(self, x, w, b):
        x = (x - np.mean(x, axis=1, keepdims=True)) / np.std(x, axis=1, keepdims=True)
        x = x * w + b
        return x

    #链接[cls] token的输出层
    def pooler_output_layer(self, x):
        x = np.dot(x, self.pooler_dense_weight.T) + self.pooler_dense_bias
        x = np.tanh(x)
        return x

    #最终输出
    def forward(self, x):
        x = self.embedding_forward(x)
        sequence_output = self.all_transformer_layer_forward(x)
        pooler_output = self.pooler_output_layer(sequence_output[0])
        return sequence_output, pooler_output


#自制
db = DiyBert(state_dict)
total_params = sum(p.numel() for p in bert.parameters())
trainable_params = sum(p.numel() for p in bert.parameters() if p.requires_grad)
non_trainable_params = sum(p.numel() for p in bert.parameters() if not p.requires_grad)
print("代码统计可训练参数量为：{}".format(trainable_params))
diy_sequence_output, diy_pooler_output = db.forward(x)
#torch
torch_sequence_output, torch_pooler_output = bert(torch_x)

# print(diy_sequence_output)
# print(torch_sequence_output)

# print(diy_pooler_output)
# print(torch_pooler_output)