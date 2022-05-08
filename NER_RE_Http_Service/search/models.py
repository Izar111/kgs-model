from django.db import models
from . import views

import sys
import ast

sys.path.append("/home/kgs/Project-KGS/NER_RE")

from ner_re import main as ner_re

# Create your models here.
def main(request):
    sentence = ''
    link = '' #request.GET['link']

    if 'sentence' in request.GET.dict():
        sentence = ast.literal_eval(request.GET['sentence'])
    if 'link' in request.GET.dict():
        link = ast.literal_eval(request.GET['link'])

    print("sentence : ", sentence, ", type : ", type(sentence), "\nlink : ", link, ", type : ", type(link))
    print(">>>Start NER-RE===============")
    res = ner_re(sentence, link)
    print(">>>End NER-RE=================")

    return views.result(res)
