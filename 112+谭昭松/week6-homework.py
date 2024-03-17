import torch
import numpy as np
import torch.nn as nn
from transformers import BertModel

bert = BertModel.from_pretrained(r"D:\\NLP+CV\\第六周\\bert-base-chinese",
                        output_hidden_states=True, output_attentions=True)

for name, param in bert.named_parameters():
    print(name, param.shape)

# embedding layer
# embeddings.word_embeddings.weight torch.Size([21128, 768])
# embeddings.position_embeddings.weight torch.Size([512, 768])
# embeddings.token_type_embeddings.weight torch.Size([2, 768])
# embeddings.LayerNorm.weight torch.Size([768])
# embeddings.LayerNorm.bias torch.Size([768])
 
# encoder.layer.0.attention.self.query.weight torch.Size([768, 768])
# encoder.layer.0.attention.self.query.bias torch.Size([768])
# encoder.layer.0.attention.self.key.weight torch.Size([768, 768])
# encoder.layer.0.attention.self.key.bias torch.Size([768])
# encoder.layer.0.attention.self.value.weight torch.Size([768, 768])
# encoder.layer.0.attention.self.value.bias torch.Size([768])

# encoder.layer.0.attention.output.dense.weight torch.Size([768, 768])
# encoder.layer.0.attention.output.dense.bias torch.Size([768])

# encoder.layer.0.attention.output.LayerNorm.weight torch.Size([768])
# encoder.layer.0.attention.output.LayerNorm.bias torch.Size([768])
    
# encoder.layer.0.intermediate.dense.weight torch.Size([3072, 768])
# encoder.layer.0.intermediate.dense.bias torch.Size([3072])
    
# encoder.layer.0.output.dense.weight torch.Size([768, 3072])
# encoder.layer.0.output.dense.bias torch.Size([768])
    
# encoder.layer.0.output.LayerNorm.weight torch.Size([768])
# encoder.layer.0.output.LayerNorm.bias torch.Size([768])
    
# pooler.dense.weight torch.Size([768, 768])
# pooler.dense.bias torch.Size([768])

#vocab_size = 211128
#max_sequence_length = 512
#hidden_size = 768
#center_size = 4x768 = 3072

#Embedding层 211128x768+512x768+2x768+768+768
#encoder层   (768x768+768)x3+(768x768+768)+(768+768)+(3072x768+3072)+(768x3072+768)+(768+768)
#pooling层   768x768+768
#sum         24301056
total = sum(p.numel() for p in bert.parameters())
print("total param:",total)
# 24301056