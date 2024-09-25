from django.contrib import admin
from django.urls import path
from . import views
from learning_app.views import video_stream,practice

urlpatterns = [
    path('practice/',views.practice,name='practice'),
    path('video/', video_stream, name='video_stream1'),
    path('alphabets/',views.alphabets,name='alphabets'),
    path('numbers/',views.numbers,name='numbers'),
    path('words/',views.words,name='words'),
]
