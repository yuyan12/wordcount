from django.contrib import admin
from django.urls import path
from . import function

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', function.home),
    path('count/', function.count),
    path('about/', function.about),
]
