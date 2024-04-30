

"""
URL configuration for insuranceproject project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from django.conf import settings
from django.conf.urls.static import static
from userapp import views as user
from adminapp import views as main



urlpatterns = [
    path('ADMIN/', admin.site.urls),
    path('',user.user_index,name='user_index'),
    path('user/contact',user.user_contact,name='user_contact'), 
    path('user/about',user.user_about,name='user_about'), 
    path('user/services',user.user_services,name='user_services'), 
    path('user/user_login',user.user_login,name='user_login'), 
    path('user/admin_login',user.admin_login,name='admin_login'),
    path('user/registration',user.user_registration,name='user_registration'),
    path('user/dashboard',user.user_dashboard ,name='user_dashboard'),
    path('user/myprofile',user.user_myprofile ,name='user_myprofile'),
    path('user/detect',user.user_detect ,name='user_detect'),
    path('user/detectresult',user.user_detectresult ,name='user_detectresult'),
    path('user/otp',user.user_otp ,name='user_otp'),
    path('user/feedback',user.user_feedback ,name='user_feedback'),    



    path('admin/index',main.admin_index ,name='admin_index'),
    path('admin/pendingusers',main.admin_pendingusers ,name='admin_pendingusers'),
    path('admin/manageusers',main.admin_manageusers ,name='admin_manageusers'),
    path('admin/upload',main. admin_upload ,name='admin_upload'),
     path('admin_dataset_btn',main.admin_dataset_btn, name ='admin_dataset_btn'),
 
    path('admin/view',main. admin_view ,name='admin_view'),
    path('admin/viewdetails', main.admin_viewdetails,name='admin_viewdetails'),  

    path('admin/svm',main.admin_svmalgorithm ,name='admin_svmalgorithm'),
    path('admin/knn',main.admin_knn ,name='admin_knn'),
    path('admin/logistic',main.admin_logistic ,name='admin_logistic'),
    path('admin/desiciontree',main.admin_desiciontree ,name='admin_desiciontree'),
    path('admin/gradient',main.admin_gradientboost ,name='admin_gradientboost'),
    path('admin/ada',main.admin_adaboost ,name='admin_adaboost'),
    path('admin/xg',main.admin_xgboost ,name='admin_xgboost'),
    path('admin/forest',main.admin_forestalgorithm ,name='admin_forestalgorithm'),

    path('admin/graph',main.admin_graph ,name='admin_graph'),
    path('admin-rejectbtn/<int:x>',main.admin_reject_btn, name='admin_reject_btn'),
    path('admin-acceptbtn/<int:x>',main.admin_accept_btn, name='admin_accept_Btn'),
    path('admin-change-status/<int:id>',main.Change_Status, name ='change_status'),
    path('admin-delete/<int:id>',main.Delete_User, name ='delete_user'),
    path('delete-dataset/<int:id>',main.delete_dataset, name = 'delete_dataset'), 
    path('KNN_btn', main.KNN_btn, name='KNN_btn'),
    path('KNN_btn', main.KNN_btn, name='KNN_btn'),
    path('svm_btn', main.svm_btn, name='svm_btn'),
    path('decision_btn', main.decision_btn, name='decision_btn'),
    path('random_btn',main.random_btn, name='random_btn'),
    path('gradient_btn', main.gradient_btn, name='gradient_btn'),
    path('ada_btn', main.ada_btn, name='ada_btn'),
    path('xg_btn', main.xg_btn, name='xg_btn'),
    path('logistic_btn',main.logistic_btn, name='logistic_btn'),
    path('admin/sentiments',main.admin_feedbacksentiments,name='admin_sentiments'),
    path('admin/feedbackgraph',main.admin_feedebackgraph,name='admin_feedbackgraph'),
    path('admin/feedback', main.admin_feedback,name='admin_feedback'),
     path('logout_btn',main.adminlogout, name='adminlogout'),
    
  






]+static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
