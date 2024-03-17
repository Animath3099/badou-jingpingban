### 正负样本对

self.loss(vector1, vector2, target.squeeze())

 __main__ - INFO - epoch 10 begin
 __main__ - INFO - epoch average loss: 0.320749
 __main__ - INFO - 开始测试第10轮模型效果：
 __main__ - INFO - 预测集合条目总量：464
 __main__ - INFO - 预测正确条目：417，预测错误条目：47
 __main__ - INFO - 预测准确率：0.898707

## cosine_triplet_loss

self.cosine_triplet_loss(vector1, vector2, vector3, None)

 __main__ - INFO - epoch 10 begin
 __main__ - INFO - epoch average loss: 0.056814
__main__ - INFO - 开始测试第10轮模型效果：
 __main__ - INFO - 预测集合条目总量：464
 __main__ - INFO - 预测正确条目：418，预测错误条目：46
 __main__ - INFO - 预测准确率：0.900862
 __main__ - INFO - --------------------



## 小结

使用cosine_triplet_loss对模型性能的提升并不明显