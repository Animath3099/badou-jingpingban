import torch.nn as nn


class Fast_text(nn.Module):
    def __init__(self,word_dim,vocal_size,sentence_length):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=vocal_size, embedding_dim=word_dim)
        self.pool=nn.AvgPool1d(kernel_size=sentence_length)
        self.fc = nn.Linear(word_dim, word_dim*2)
        self.fc2=nn.Linear(word_dim*2,2)
        self.softmax=nn.Softmax(dim=2)
        self.loss=nn.CrossEntropyLoss()

    def computeLoss(self, y_pred, y_true):
        y_pred=y_pred.view(-1,2)

        return self.loss(y_pred, y_true.long())

    def forward(self,x):

        x=self.embedding(x)
        x = x.transpose(1, 2)
        x=self.pool(x)
        x = x.transpose(1, 2)
        x=self.fc(x)
        x=self.fc2(x)
        x=self.softmax(x)
        return x


class  LSTM(nn.Module):
    def __init__(self,vocab_size,word_dim,sentence_length):
        super().__init__()
        self.vocab_size=vocab_size
        self.word_dim=word_dim
        self.sentence_length=sentence_length

        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=word_dim)
        self.lstm=nn.LSTM(input_size=word_dim,hidden_size=word_dim*2,num_layers=1,batch_first=True)
        self.fc = nn.Linear(word_dim*2, 2)
        self.softmax=nn.Softmax(dim=2)
        self.loss=nn.CrossEntropyLoss()
        self.pool=nn.AvgPool1d(kernel_size=sentence_length)



    def computeLoss(self,y_pred,y_true):
        y_pred = y_pred.view(-1, 2)
        return self.loss(y_pred,y_true.long())


    def forward(self,x):
        x=self.embedding(x)
        x,_=self.lstm(x)
        x=x.transpose(1,2)
        x=self.pool(x)
        x=x.transpose(1,2)
        x=self.fc(x)
        x=self.softmax(x)

        return x

class CNN(nn.Module):
    def __init__(self,vocab_size,word_dim,sentence_length):
        super().__init__()
        self.vocab_size=vocab_size
        self.word_dim=word_dim
        self.sentence_length=sentence_length

        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=word_dim)
        self.conv1 = nn.Conv1d(in_channels=word_dim, out_channels=word_dim, kernel_size=3)
        self.pool = nn.AvgPool1d(kernel_size=28)
        self.fc=nn.Linear(self.word_dim,2)
        self.softmax=nn.Softmax(dim=2)
        self.loss=nn.CrossEntropyLoss()
    def computeLoss(self,y_pred,y_true):
        y_pred = y_pred.view(-1, 2)
        return self.loss(y_pred, y_true.long())

    def forward(self,x):
        x=self.embedding(x)

        #---- batch_size, word_dim, sentence_length----
        x=x.transpose(1,2)
        x=self.conv1(x)
        #batch_size,word_dim,1
        x=self.pool(x)

        #batch 1,word_dim
        x=x.transpose(1,2)
        x=self.fc(x)
        x=self.softmax(x)
        return x

class Bert(nn.Module):
    def __init__(self):
        super().__init__()
