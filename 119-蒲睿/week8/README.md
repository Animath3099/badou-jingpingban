# Week8 作业

model.py  
```python
def forward(self, sentence1, sentence2=None, sentence3=None):  # target相当于是label
        #同时传入两个句子
        if sentence2 is not None and sentence3 is not None:
            v1 = self.sentence_encoder(sentence1)
            v2 = self.sentence_encoder(sentence2)
            v3 = self.sentence_encoder(sentence3)
            return self.cosine_triplet_loss(v1, v2, v3)
        #单独传入一个句子时，认为正在使用向量化能力
        else:
            return self.sentence_encoder(sentence1)
```

loader.py
```python
    def random_train_sample(self):
        standard_question_index = list(self.knwb.keys())
        '''
        if self.triplet:
            postive = random.choice(standard_question_index)
            if len(self.knwb[postive]) < 2:
                return self.random_train_sample()
            else:
                a, p = random.sample(self.knwb[postive], 2)
            negative = random.choice(standard_question_index.remove(postive))
            n = random.sample(self.knwb[negative], 1)
            return a, p, n
        '''
        p, n = random.sample(standard_question_index, 2)
        if len(self.knwb[p]) < 2:
            s1 = s2 = self.knwb[p][0]
        else:
            s1, s2 = random.sample(self.knwb[p], 2)
        s3 = random.sample(self.knwb[n], 1)
        return [s1, s2, s3]
```
main.py  
```python
for index, batch_data in enumerate(train_data):
            optimizer.zero_grad()
            if cuda_flag:
                batch_data = [d.cuda() for d in batch_data]
            
            input_id1 = batch_data[0]
            input_id2 = batch_data[1]
            input_id3 = batch_data[2][0]
            #输入变化时这里需要修改，比如多输入，多输出的情况
            loss = model(input_id1, input_id2, input_id3)
            train_loss.append(loss.item())
            # if index % int(len(train_data) / 2) == 0:
            #     logger.info("batch loss %f" % loss)
            loss.backward()
            optimizer.step()
        logger.info("epoch average loss: %f" % np.mean(train_loss))
        evaluator.eval(epoch)
```


