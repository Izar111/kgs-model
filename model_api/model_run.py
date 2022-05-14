import sys
import ast

sys.path.append("/home/kgs/Desktop/kgs-model/NER_RE")

from run import main as ner_re

def err_func(str):
    return {'error':str}
     
def main(request):

    # model output stream change 
    keep_stdout , sys.stdout = sys.stdout, open('model_log.log','w')
    # call_model arg check 
    print("req : ", request, ", type : ", type(request))

    sentence = ''
    link = '' #request.GET['link']
    inputs = ast.literal_eval(request) # str -> dict
    # model input check
    print("inputs : ", inputs, ", type : ", type(inputs))
    # diction value extraction
    if 'sentence' in inputs:
        sentence = inputs['sentence']
    if 'link' in inputs:
        link = inputs['link']

    #if len(sentence) != len(link)
    #    return err_func("sentence and link is no match !!")

    print("sentence : ", sentence, ", type : ", type(sentence), "\nlink : ", link, ", type : ", type(link))
    print(">>>Start NER-RE===============")
    # run ner_re model running
    res = ner_re(sentence, link)

    print(">>>End NER-RE=================")
    # model output stream revert
    sys.stdout.close()
    sys.stdout = keep_stdout
    # output stream refresh
    sys.stdout.flush()
    # give model result to server
    print(res)
    
    return res

# if no input => read origin input.txt
def no_arg():
    print(">>>Start NER-RE===============")
    # run ner_re model running
    res = ner_re(sentence, link)

    print(">>>End NER-RE=================")
    # model output stream revert
    sys.stdout.close()
    sys.stdout = keep_stdout
    # output stream refresh
    sys.stdout.flush()
    # give model result to server
    print(res)
    
    return res


if __name__ == '__main__':
    #print(sys.argv)
    if sys.argv[1] == 'undefined': # if no argv
        no_arg()
    else :
        main(sys.argv[1])
