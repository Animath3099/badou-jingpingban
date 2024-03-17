import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib



class TorchModel(nn.Module):
    def __init__(self, vector_dim, sentence_length, vocab):
        super(TorchModel, self).__init__()
        self.embedding = nn.Embedding(len(vocab), vector_dim)
        self.pool = nn.AvgPool1d(sentence_length)
        self.structrue = nn.Linear(vector_dim, 1)
        self.activation = torch.sigmoid
        self.loss = nn.functional.mse_loss

    def forward(self, x, y=None):
        x = self.embedding(x)
        x = x.transpose(1, 2)
        x = self.pool(x)
        x = x.squeeze()
        x = self.structrue(x)
        y_pred = self.activation(x)
        if y is not None:
            return self.loss(y_pred, y)
        else:
            return y_pred



def build_vocab():
    chars = "abcdefghijklmnopqrstuvwxyz"
    vocab = {"pad": 0}
    for index, char in enumerate(chars):
        vocab[char] = index + 1
    vocab['unk'] = len(vocab)
    return vocab


def build_sample(vocab, sentence_length):
    x = [random.choice(list(vocab.keys())) for _ in range(sentence_length)]
    if set("abc") & set(x):
        y = 1
    else:
        y = 0
    x = [vocab.get(word, vocab['unk']) for word in x]
    return x, y



def build_dataset(sample_length, vocab, sentence_length):
    X = []
    Y = []
    for i in range(sample_length):
        x, y = build_sample(vocab, sentence_length)
        X.append(x)
        Y.append([y])
    return torch.LongTensor(X), torch.FloatTensor(Y)



def build_model(char_dim, sentence_length, vocab):
    model = TorchModel(char_dim, sentence_length, vocab)
    return model



def train():
    epoch_num = 20
    batch_size = 20
    train_sample = 500
    char_dim = 20
    sentence_length = 6
    learning_rate = 0.01

    vocab = build_vocab()
    model = build_model(char_dim, sentence_length, vocab)
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for epoch in range(epoch_num):
        model.train()
        for batch in range(int(train_sample / batch_size)):
            x, y = build_dataset(batch_size, vocab, sentence_length)
            optim.zero_grad()
            loss = model.forward(x, y)
            print(type(loss))
            loss.backward()
            optim.step()
    torch.save(model.state_dict(), "model.pth")
    with open("vocab.json", "w") as f:
        f.write(json.dumps(vocab, ensure_ascii=False, indent=2))



def predict(model_path, vocab_path, input_strings):
    char_dim = 20
    sentence_length = 6
    vocab = json.load(open(vocab_path, "r", encoding="utf-8"))
    model = build_model(char_dim, sentence_length, vocab)
    model.load_state_dict(torch.load(model_path))
    x = []
    for input_string in input_strings:
        x.append([vocab[char] for char in input_string])
    model.eval()
    with torch.no_grad():
        result = model.forward(torch.LongTensor(x))
    for i, input_string in enumerate(input_strings):
        print("输入: %s, 预测类别: %d, 概率值:%f" % (input_string, round(float(result[i])), result[i]))


if __name__ == "__main__":
    # train()
    test_strings = ["sffmws", "ldncha", "pemnsdj", "mdgxiq"]
    predict("model.pth", "vocab.json", test_strings)
