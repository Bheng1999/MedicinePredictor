
from django.contrib import admin
from django.urls import path
from core import views
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.index, name='index'),
    path('login/', views.loginView, name='login'),
    path("register/", views.registerView, name="register"),

    path('dashboard/', views.dashboardView, name='dashboard'),
    path('dashboard/logout/', views.logoutView, name='logout'),

    path('upload/', views.uploadView, name="uploadPage"),
    path('upload/uploadCsv/', views.uploadCsv, name="uploadCsv"),

    path('train/', views.trainView, name="trainPage"),
    path('train/start/', views.trainModel, name="trainModel"),

    path('predict/', views.predictView, name="predictPage"),
    path('predict/makePrediction/', views.makePrediction, name="makePrediction"),

    path('result/', views.resultView, name="resultPage"),


    path('manage-users/', views.manageUserView, name="manageUsersPage"),
    path('manage-users/add/', views.addUserView, name='add_user'),
    path('manage-users/edit/', views.editUserView, name='edit_user'),
    path('manage-users/delete/', views.deleteUserView, name='delete_user'),


    path('profile/', views.profileView, name='profilePage'),
    path('profile/update/', views.profileChange, name='profileUpdate'),
    path('profile/change-password/', views.passwordChange, name='passwordChange'),
    

    path('search/', views.searchView, name='searchPage'),
    path('search/results/', views.searchMed, name='searchMed'),
    
    

]+ static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
