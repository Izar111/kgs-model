import sys
import ast

sys.path.append("/home/kgs/Desktop/kgs-model/NER_RE")

from run import main as ner_re

def err_func(str):
    return {'error':str}
    
def main(request):

    keep_stdout , sys.stdout = sys.stdout, open('model_log.log','w')

    print("req : ", request, ", type : ", type(request))

    sentence = ''
    link = '' #request.GET['link']
    inputs = ast.literal_eval(request)

    print("inputs : ", inputs, ", type : ", type(inputs))

    if 'sentence' in inputs:
        sentence = inputs['sentence']
    if 'link' in inputs:
        link = inputs['link']

    #if len(sentence) != len(link)
    #    return err_func("sentence and link is no match !!")

    print("sentence : ", sentence, ", type : ", type(sentence), "\nlink : ", link, ", type : ", type(link))
    print(">>>Start NER-RE===============")
    
    res = ner_re(sentence, link)

    print(">>>End NER-RE=================")

    sys.stdout.close()
    sys.stdout = keep_stdout

    sys.stdout.flush()

    print(res)
    
    return res


if __name__ == '__main__':
    if len(sys.argv)<2:
        main()
    else :
        main(sys.argv[1])
