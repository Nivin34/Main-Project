from django.contrib import admin
from django.urls import path
from . import views
from sign_app.views import *

urlpatterns = [
    path('sign/',views.sign,name='sign'),
    path('video/', video_stream, name='video_stream'),
    
]