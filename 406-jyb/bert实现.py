import torch
import math
import numpy as np
from transformers import BertModel

bert = BertModel.from_pretrained(r"E:\python\bert_file", return_dict=False)
state_dict = bert.state_dict()
bert.eval()

#假设x是输入
x=np.array([1,2,3,4])
torch_x = torch.LongTensor([x])  #pytorch形式输入

#softmax
def softmax(x):

    #keepdims=True,不会减少维度。比如(3,1,5)不会减少到(3,5)
    return np.exp(x)/np.sum(np.exp(x),axis=-1,keepdims=True)

def gelu(x):
    return 0.5 * x * (1 + np.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * np.power(x, 3))))

class Bert():
    #将预训练好的权重字典放进来
    def __init__(self,state_dict):

        #使用多头注意力机制
        self.num_head=12
        self.hidden_size=768

        #transformer层数
        self.num_layer=1
        self.load_weight(state_dict)

    def load_weight(self,state_dict):
        # embedding部分
        self.word_embeddings = state_dict["embeddings.word_embeddings.weight"].numpy()
        self.position_embeddings = state_dict["embeddings.position_embeddings.weight"].numpy()
        self.token_type_embeddings = state_dict["embeddings.token_type_embeddings.weight"].numpy()
        self.embeddings_layer_norm_weight = state_dict["embeddings.LayerNorm.weight"].numpy()
        self.embeddings_layer_norm_bias = state_dict["embeddings.LayerNorm.bias"].numpy()
        self.transformer_weights = []

        for i in range(self.num_layer):
            q_w = state_dict["encoder.layer.%d.attention.self.query.weight" % i].numpy()
            q_b = state_dict["encoder.layer.%d.attention.self.query.bias" % i].numpy()
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
            self.transformer_weights.append([q_w, q_b, k_w, k_b, v_w, v_b, attention_output_weight, attention_output_bias,
                                             attention_layer_norm_w, attention_layer_norm_b, intermediate_weight, intermediate_bias,
                                             output_weight, output_bias, ff_layer_norm_w, ff_layer_norm_b])
        # pooler层
        self.pooler_dense_weight = state_dict["pooler.dense.weight"].numpy()
        self.pooler_dense_bias = state_dict["pooler.dense.bias"].numpy()

    def get_embedding(self,embedding_matrix,x):

        #相当于 max_length个one hot的向量与embedding矩阵相乘。
        #或者理解为按照x中的值查找embedding中对应的向量
        return np.array([embedding_matrix[index] for index in x])


    def embedding_forward(self,x):
        '''
        三个embedding层的参数两相同，都是vocab_size*embedding_dim----------------也就是21130*128=16227840,一千六百多万个参数
        总参数=48683520
        :param x:
        :return:
        '''
        # x shape [max_length]
        #相当于将一句话embedding为了一个矩阵。
        #矩阵的大小是sentence_length*embeddings_dim.
        #这里embedding层的参数量为：vocab_size*embedding_dim----------------也就是21130*128=16227840,一千六百多万个参数
        we=self.get_embedding(self.word_embeddings,x)


        #position_embedding,这里相当于输入的句子有4个字，所以取权重矩阵的前4个位置,也就是下标[0,1,2,3]
        pe=self.get_embedding(self.position_embeddings,np.array([i for i in range(len(x))]))

        ##segment embedding,也就是句子种类embedding。一个句子对应一个下标。这里只输入了一个句子。所以都取embedding的第0个向量
        se=self.get_embedding(self.token_type_embeddings,np.array([0]*len(x)))

        embedding = we + pe + se
        embedding = self.layer_norm(embedding, self.embeddings_layer_norm_weight,
                                    self.embeddings_layer_norm_bias)  # shpae: [max_len, hidden_size]
        return embedding

    def all_transformers_layer_forward(self,x):
        for i in range(self.num_layer):
            x = self.single_transformer_layer_forward(x, i)
        return x

    #单层的transformers计算
    def single_transformer_layer_forward(self,x,layer_index):
        weights = self.transformer_weights[layer_index]
        q_w, q_b, \
            k_w, k_b, \
            v_w, v_b, \
            attention_output_weight, attention_output_bias, \
            attention_layer_norm_w, attention_layer_norm_b, \
            intermediate_weight, intermediate_bias, \
            output_weight, output_bias, \
            ff_layer_norm_w, ff_layer_norm_b = weights

        #self-attention层
        attention_output = self.self_attention(x,
                                               q_w, q_b,
                                               k_w, k_b,
                                               v_w, v_b,
                                               attention_output_weight, attention_output_bias,
                                               self.num_head,
                                               self.hidden_size)

        #归一化层，使用残差机制
        x = self.layer_norm(x + attention_output, attention_layer_norm_w, attention_layer_norm_b)

        # feed forward层
        feed_forward_x = self.feed_forward(x,
                                           intermediate_weight, intermediate_bias,
                                           output_weight, output_bias)

        x = self.layer_norm(x + feed_forward_x, ff_layer_norm_w, ff_layer_norm_b)
        return x

    def self_attention(self,x,q_w,q_b,k_w,k_b,v_w, v_b, attention_output_weight, attention_output_bias,num_attention_heads, hidden_size):

        #q_w的参数量为:768*768
        #那么三个权重和三个偏置的总参数量为:3538944,三百多万
        #x shape: sentence_length, hidden_size


        Q=np.dot(x,q_w.T)+q_b
        K=np.dot(x,k_w.T)+k_b
        V=np.dot(x,v_w.T)+v_b

        #多头注意力机制的准备，计算每头有是多少维。比如768维分12头每头就是64维
        attention_head_size=int(hidden_size / num_attention_heads)
        q=self.transpose_for_scores(Q,num_attention_heads,attention_head_size)
        k=self.transpose_for_scores(K,num_attention_heads,attention_head_size)
        v=self.transpose_for_scores(V,num_attention_heads,attention_head_size)

        #[num_attention_heads, max_len, attention_head_size]*[num_attention_heads, attention_head_size, max_len]
        #qk.shape = num_attention_heads, max_len, max_len]
        Q_K=np.matmul(q,k.swapaxes(1,2))
        Q_K /= np.sqrt(attention_head_size)
        Q_K=softmax(Q_K)

        #shape:[num_attention_heads, max_len, max_len]*[num_attention_heads, max_len, attention_head_size]
        #QKV.shape = num_attention_heads, max_len, attention_head_size
        QKV = np.matmul(Q_K, v)

        qkv = QKV.swapaxes(0, 1).reshape(-1, hidden_size)
        # attention.shape = max_len, hidden_size

        #对QKV再输入线性层
        attention = np.dot(qkv, attention_output_weight.T) + attention_output_bias
        return attention

    #多头机制
    def transpose_for_scores(self, x, num_attention_heads,attention_head_size):
        max_len,hidden_size=x.shape
        x=x.reshape(max_len, num_attention_heads, attention_head_size)

        #变成[12.4,768]
        x = x.swapaxes(1, 0)  # output shape = [num_attention_heads, max_len, attention_head_size]
        return x

    def layer_norm(self, x, w, b):
        x = (x - np.mean(x, axis=1, keepdims=True)) / np.std(x, axis=1, keepdims=True)
        x = x * w + b
        return x

    def feed_forward(self,x,intermediate_weight, intermediate_bias, output_weight, output_bias,):
        #参数量:max_len*intermediate_size*2

        # output shpae: [max_len, intermediate_size]
        x = np.dot(x, intermediate_weight.T) + intermediate_bias
        x = gelu(x)
        # output shpae: [max_len, hidden_size]
        x = np.dot(x, output_weight.T) + output_bias
        return x

    def pooler_output_layer(self, x):
        x = np.dot(x, self.pooler_dense_weight.T) + self.pooler_dense_bias
        x = np.tanh(x)
        return x
    def forward(self, x):


        x = self.embedding_forward(x)

        #按照 [sen_len,hidden]送入自注意力模块，输出还是max_len,hidden的形状
        sequence_output = self.all_transformers_layer_forward(x)

        pooler_output = self.pooler_output_layer(sequence_output[0])
        return sequence_output, pooler_output

db = Bert(state_dict)
diy_sequence_output, diy_pooler_output = db.forward(x)
#torch
torch_sequence_output, torch_pooler_output = bert(torch_x)

print(diy_sequence_output)
print(torch_sequence_output)