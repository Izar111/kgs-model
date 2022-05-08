from django.http import HttpResponse

def result(parameters):
    printVal = "<table border='1'>"
    for param in parameters:
        printVal += "<tr><td>단어1 : "+param["word1"]+"["+param["ner1"]+"]</td>"+"<td>단어2 : "+param["word2"]+"["+param["ner2"]+"]</td>"+"<td>관계 :" + param["re"] + "</td></tr>"
        #print(param)
    printVal += "</table>"
    return HttpResponse(printVal)

def input_test(sentence,link):
    printParameter = "sentence : "+sentence+", link : "+link
    return HttpResponse(printParameter)

def main(request):
    print(request.GET)
    return HttpResponse('test')