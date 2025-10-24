from django.urls import path
from . import views

urlpatterns = [
    path('', views.signup_view, name='signup'),
    path('login/', views.login_view, name='login'),
    path('home/', views.home_view, name='chat_home'),  
    path('chat/', views.chat_view, name='chat'),
    path('documents/', views.upload_document, name='documents'),
    path('ask-doc/', views.ask_doc_question, name='ask_doc_question'),
    path("logout/", views.logout_view, name="logout"),
    path("delete_all_chats/", views.delete_all_chats, name="delete_all_chats"),
    path("upload/", views.upload_document, name="upload_document"),
    path('ask-question/', views.ask_doc_question, name='ask_question'),
    path("documents/download/<int:doc_id>/", views.download_document, name="download_document"),
    path('documents/delete/<int:doc_id>/', views.delete_document, name='delete_document'),
]
