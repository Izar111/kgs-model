import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from kobert.utils import get_tokenizer
from kobert.pytorch_kobert import get_pytorch_kobert_model
from BERT_classifier import BERTClassifier
from transformers import AutoModelForTokenClassification
from RE_predict import predict as RE_pdt

from NER_Predict import predict as NER_pdt

from postprocessor import Process_Kkma
from postprocessor import Process_Mecab as josa_process

import sys

Porject_path = "/home/kgs/Project-KGS/NER_RE_20220420"

NER_path = Porject_path+"/model/NER"
RE_path = Porject_path+"/model/RE"

def RE():
    device = torch.device("cuda:0")
    bertmodel, vocab = get_pytorch_kobert_model()
    # bert 모델 불러오기
    RE_model = BERTClassifier(bertmodel, dr_rate=0.5).to(device)
    RE_model.load_state_dict(torch.load(RE_path + '/model20.pt'))

    return RE_model, vocab

def main(inputs=None,links=None):
    model_res = []
    if inputs is None:
        sentence, ner_res = NER_pdt(model_dir=NER_path, input_file_dir="./In-Out/input_NER.txt")
    else:
        sentence, ner_res = NER_pdt(model_dir=NER_path, lines=inputs)

    josa_divide = []
    for sen in sentence:
        josa_divide.append(josa_process(sen))
    sentence = josa_divide
    print(sentence)

    RE_model, RE_vocab = RE()
    tokenizer = get_tokenizer()
    for i in range(len(sentence)):
        print(sentence[i])

        word1 = sentence[i].split('<e1>')[1].split('</e1>')[0]
        word2 = sentence[i].split('<e2>')[1].split('</e2>')[0]

        ner1, ner2, link = ner_res[i]
        re_res = RE_pdt(RE_model, tokenizer, RE_vocab, sentence[i])
        print("word1", word1, "ner1", ner1, "word2", word2, "ner2", ner2, "re", re_res)

        if links is None:
            model_res.append({"word1": word1, "ner1": ner1, "word2": word2, "ner2": ner2, "re": re_res, "link": link})
        else:
            model_res.append({"word1": word1, "ner1": ner1, "word2": word2, "ner2": ner2, "re": re_res, "link": link})

    return model_res

if __name__ == "__main__":
	main()
