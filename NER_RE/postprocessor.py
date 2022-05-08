from konlpy.tag import Kkma
from konlpy.tag import Mecab

kkma = Kkma()
mecab = Mecab('/home/kgs/Project-KGS/NER-RE/mecab-ko-dic')

def Process_Kkma(text):
    tags = ['<e1>','</e1>','<e2>','</e2>']
    loc = [0, 0, 0, 0]
    for i in range(0,4):
        loc[i] = text.find(tags[i])
    entity = [kkma.pos(text[loc[0]+4:loc[1]]), kkma.pos(text[loc[2]+4:loc[3]])]
    for i in range(0,2):
        for j in range(0,len(entity[i])):
            tmp = entity[i][len(entity[i])-j-1][1]
            if tmp.find('JK') != -1 or  tmp == 'JX' or tmp == 'JC' :
                continue
            elif j > 0 :
                text = text.replace(tags[2*i+1],'')
                text = text[:loc[2*i+1]-j] + tags[2*i+1] + text[loc[2*i+1]-j:]
            break

    return text


def Process_Mecab(original_sentence, need_change_word=None):
    original = mecab.pos(original_sentence)
    josa = ['JKS', 'JKC', 'JKG', 'JKO', 'JKB', 'JKV', 'JKQ', 'JX', 'JC']
    temp = []
    flag = False

    index = 0
    while index < len(original):
        if original[index][0]=='<':
            flag = True
        elif original[index][0]=='</':
            flag = False
            count = 4
        for i in temp:
            original.insert(index+count, i)
            count += 1
        temp = []
        if flag and original[index][1] in josa:
            while flag and original[index][1] in josa:
                temp.append(original[index])
                del original[index]
            continue
        index+=1

    temp_sentence = ''
    for i in range(len(original)):
        temp_sentence += original[i][0]
    index = 0
    answer_sentence = ''
    for i in original_sentence.split(' '):
        answer_sentence += temp_sentence[index:index+len(i)] + ' '
        index = index+len(i)

    if need_change_word is not None and need_change_word is True:
        return answer_sentence
    else:
        return answer_sentence

def josa_process(original_sentence):
  original = mecab.pos(original_sentence)
  josa = ['JKS', 'JKC', 'JKG', 'JKO', 'JKB', 'JKV', 'JKQ', 'JX', 'JC']
  temp = []
  flag = False
  index = 0
  while 1:
    if index==len(original):
      break
    if original[index][0]=='<':
      flag = True
    elif original[index][0]=='</':
      flag = False
      count = 4
      for i in temp:
        original.insert(index+count, i)
        count += 1
      temp = []
    if flag and original[index][1] in josa:
      while flag and original[index][1] in josa:
        temp.append(original[index])
        del original[index]
      continue
    index+=1

  temp_sentence = ''
  for i in range(len(original)):
    temp_sentence += original[i][0]

  index = 0
  answer_sentence = ''
  for i in original_sentence.split(' '):
    answer_sentence += temp_sentence[index:index+len(i)] + ' '
    index = index+len(i)
  return answer_sentence