from django.urls import path
from . import views


urlpatterns = [
    path('', views.login_view, name='login'),
    path('register/', views.register, name='register'),
    path('logout/', views.logout_view, name='logout'),
    path('home/', views.home, name='home'),
    path('feature/', views.feature, name='feature'),
    path('about/', views.about, name='about'),
    path('contact/', views.contact, name='contact'),
    path('meeting/',views.videocall, name='meeting'),
    path('dashboard/',views.dashboard, name='dashboard'),
    path('join/',views.join_room, name='join_room'),
    path('chat/', views.external_redirect_view, name='external_redirect'),

    
]
