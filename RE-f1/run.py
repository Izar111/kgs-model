import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import gluonnlp as nlp
import numpy as np
from tqdm import tqdm, tqdm_notebook
from kobert.utils import get_tokenizer
from kobert.pytorch_kobert import get_pytorch_kobert_model
from transformers import AdamW
from transformers.optimization import get_cosine_schedule_with_warmup
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
import pandas as pd
import json

device = torch.device("cuda:0")
bertmodel, vocab = get_pytorch_kobert_model()

with open("klue-re-v1.1_train.json") as t:
  train_data = json.load(t)
with open("klue-re-v1.1_dev.json") as d:
  dev_data = json.load(d)
with open("relation_list.json") as r:
  relation = json.load(r)

t_data = pd.json_normalize(train_data)
d_data = pd.json_normalize(dev_data)
r_list = pd.json_normalize(relation)

print(t_data.index)
print(d_data.index)
print(r_list.index)
t_data.head()

def se(data):
  subj_s = "<e1>"
  subj_e = "</e1>"
  obj_s = "<e2>"
  obj_e = "</e2>"

  for i in range(len(data)):
    sub_josa = data.iloc[i][1][data.iloc[i][6]+1:].find(" ")
    obj_josa = data.iloc[i][1][data.iloc[i][10]+1:].find(" ")
    sub_josa_parenthesis = data.iloc[i][1][data.iloc[i][5]:].find("(")
    obj_josa_parenthesis = data.iloc[i][1][data.iloc[i][9]:].find("(")


    if sub_josa < 0 or sub_josa + len(data.iloc[i][1][:data.iloc[i][6]+1]) > len(data.iloc[i][1][:data.iloc[i][9]]) or (sub_josa_parenthesis < sub_josa and len(data.iloc[i][1][:data.iloc[i][5]])+ sub_josa_parenthesis >= len(data.iloc[i][1][:data.iloc[i][6]+1])):
      sub_josa = 0
    if obj_josa <0 or obj_josa + len(data.iloc[i][1][:data.iloc[i][10]+1]) > len(data.iloc[i][1][:data.iloc[i][5]]) or (obj_josa_parenthesis < obj_josa and len(data.iloc[i][1][:data.iloc[i][9]]) + obj_josa_parenthesis >= len(data.iloc[i][1][:data.iloc[i][10]+1])):
      obj_josa = 0
    if data.iloc[i][9] < data.iloc[i][5]: #목적어가 주어보다 먼저 나오는 경우
      sen = data.iloc[i][1][:data.iloc[i][9]] + obj_s + data.iloc[i][1][data.iloc[i][9]:data.iloc[i][10]+1+obj_josa] + obj_e + data.iloc[i][1][data.iloc[i][10]+1+obj_josa:data.iloc[i][5]] + subj_s + data.iloc[i][1][data.iloc[i][5]:data.iloc[i][6]+1+sub_josa] + subj_e + data.iloc[i][1][data.iloc[i][6]+1+sub_josa:]
    else :
      sen = data.iloc[i][1][:data.iloc[i][5]] + subj_s + data.iloc[i][1][data.iloc[i][5]:data.iloc[i][6]+1+sub_josa] + subj_e + data.iloc[i][1][data.iloc[i][6]+1+sub_josa:data.iloc[i][9]] + obj_s + data.iloc[i][1][data.iloc[i][9]:data.iloc[i][10]+1+obj_josa] + obj_e + data.iloc[i][1][data.iloc[i][10]+1+obj_josa:]
    data.sentence[i] = sen
    print( str(i) + " / " + str(len(data)))

se(t_data)
se(d_data)

data_concat = pd.concat([t_data, d_data])

encoder = LabelEncoder()
encoder.fit(data_concat['label'])
data_concat['label'] = encoder.transform(data_concat['label'])
mapping = dict(zip(range(len(encoder.classes_)), encoder.classes_))

data_list = []
for sen, label in zip(data_concat['sentence'], data_concat['label'])  :
    data = []
    data.append(sen)
    data.append(str(label))

    data_list.append(data)

from sklearn.model_selection import train_test_split
train, test = train_test_split(data_list, test_size=0.2, random_state=42)
print("train shape is:", len(train))
print("test shape is:", len(test))

class BERTDataset(Dataset):
    def __init__(self, dataset, sentence_idx, label_idx, bert_tokenizer, max_len, pad, pair):
        transform = nlp.data.BERTSentenceTransform(
            bert_tokenizer, max_seq_length=max_len, pad=pad, pair=pair)

        self.sentence = [transform([i[sentence_idx]]) for i in dataset]
        self.label = [np.int32(i[label_idx]) for i in dataset]

    def __getitem__(self, i):
        return (self.sentence[i] + (self.label[i],))

    def __len__(self):
        return (len(self.label))

max_len = 100 # 해당 길이를 초과하는 단어에 대해선 bert가 학습하지 않음
batch_size = 64
warmup_ratio = 0.1
num_epochs = 20
max_grad_norm = 1
log_interval = 200
learning_rate = 5e-5

tokenizer = get_tokenizer()
tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)

data_train = BERTDataset(train, 0, 1, tok, max_len, True, False)
data_test = BERTDataset(test, 0, 1, tok, max_len, True, False)

train_dataloader = torch.utils.data.DataLoader(data_train, batch_size=batch_size, num_workers=5)
test_dataloader = torch.utils.data.DataLoader(data_test, batch_size=batch_size, num_workers=5)


class BERTClassifier(nn.Module):
    def __init__(self,
                 bert,
                 hidden_size=768,
                 num_classes=30,  # softmax 사용 <- binary일 경우는 2
                 dr_rate=None,
                 params=None):
        super(BERTClassifier, self).__init__()
        self.bert = bert
        self.dr_rate = dr_rate

        self.classifier = nn.Linear(hidden_size, num_classes)
        if dr_rate:
            self.dropout = nn.Dropout(p=dr_rate)

    def gen_attention_mask(self, token_ids, valid_length):
        attention_mask = torch.zeros_like(token_ids)
        for i, v in enumerate(valid_length):
            attention_mask[i][:v] = 1
        return attention_mask.float()

    def forward(self, token_ids, valid_length, segment_ids):
        attention_mask = self.gen_attention_mask(token_ids, valid_length)

        _, pooler = self.bert(input_ids=token_ids, token_type_ids=segment_ids.long(),
                              attention_mask=attention_mask.float().to(token_ids.device))
        if self.dr_rate:
            out = self.dropout(pooler)
        return self.classifier(out)

model = BERTClassifier(bertmodel, dr_rate=0.5).to(device)

model.load_state_dict(torch.load("/home/kgs/Project-KGS/NER-RE/model/RE/model20.pt"))
model.eval()

# 토큰화
tokenizer = get_tokenizer()
tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)


def predict(predict_sentence):
    data = [predict_sentence, '0']
    dataset_another = [data]

    another_test = BERTDataset(dataset_another, 0, 1, tok, max_len, True, False)
    test_dataloader = torch.utils.data.DataLoader(another_test, batch_size=batch_size, num_workers=5)

    model.eval()

    for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(test_dataloader):
        token_ids = token_ids.long().to(device)
        segment_ids = segment_ids.long().to(device)

        valid_length = valid_length
        label = label.long().to(device)

        out = model(token_ids, valid_length, segment_ids)

        test_eval = []
        for i in out:
            logits = i
            logits = logits.detach().cpu().numpy()

            if np.argmax(logits) == 0:
                test_eval.append("0")
            elif np.argmax(logits) == 1:
                test_eval.append("1")
            elif np.argmax(logits) == 2:
                test_eval.append("2")
            elif np.argmax(logits) == 3:
                test_eval.append("3")
            elif np.argmax(logits) == 4:
                test_eval.append("4")
            elif np.argmax(logits) == 5:
                test_eval.append("5")
            elif np.argmax(logits) == 6:
                test_eval.append("6")
            elif np.argmax(logits) == 7:
                test_eval.append("7")
            elif np.argmax(logits) == 8:
                test_eval.append("8")
            elif np.argmax(logits) == 9:
                test_eval.append("9")
            elif np.argmax(logits) == 10:
                test_eval.append("10")
            elif np.argmax(logits) == 11:
                test_eval.append("11")
            elif np.argmax(logits) == 12:
                test_eval.append("12")
            elif np.argmax(logits) == 13:
                test_eval.append("13")
            elif np.argmax(logits) == 14:
                test_eval.append("14")
            elif np.argmax(logits) == 15:
                test_eval.append("15")
            elif np.argmax(logits) == 16:
                test_eval.append("16")
            elif np.argmax(logits) == 17:
                test_eval.append("17")
            elif np.argmax(logits) == 18:
                test_eval.append("18")
            elif np.argmax(logits) == 19:
                test_eval.append("19")
            elif np.argmax(logits) == 20:
                test_eval.append("20")
            elif np.argmax(logits) == 21:
                test_eval.append("21")
            elif np.argmax(logits) == 22:
                test_eval.append("22")
            elif np.argmax(logits) == 23:
                test_eval.append("23")
            elif np.argmax(logits) == 24:
                test_eval.append("24")
            elif np.argmax(logits) == 25:
                test_eval.append("25")
            elif np.argmax(logits) == 26:
                test_eval.append("26")
            elif np.argmax(logits) == 27:
                test_eval.append("27")
            elif np.argmax(logits) == 28:
                test_eval.append("28")
            elif np.argmax(logits) == 29:
                test_eval.append("29")

    print(test_eval)
    return test_eval[0]


guesses = []
print(len(test))
for i in range(len(test)):
    guesses.append(predict(test[i][0]))
    print(i)

labels = []
for i in range(len(test)):
  labels.append(test[i][1])

print(len(guesses))
print(len(labels))

print(f1_score(labels, guesses, average = 'micro'))
print(f1_score(labels, guesses, average = 'macro'))
print(f1_score(labels, guesses, average = 'weighted'))
print(f1_score(labels, guesses, average = None))
print(f1_score(labels, guesses))

