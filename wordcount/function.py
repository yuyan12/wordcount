#from django.http import HttpResponse
from django.shortcuts import render #传网页给用户


def home(request):
    return render(request, 'home.html')


def count(request):
    total_count = len(request.GET['text']) #text是字典形式的
    user_text = request.GET['text']
    #return render(request, 'count.html')#render可以向html中传递信息，使用字典的方式
    return render(request, 'count.html',
                 {'count': total_count, 'text':user_text})#render可以向html中传递信息，使用字典的方式
                 

def about(request):
    return render(request, 'about.html')