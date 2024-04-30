from django.shortcuts import render, redirect
from userapp.models import *
from adminapp.models import *

from django.conf import settings
from django.contrib import messages
from django.core.paginator import Paginator
import pandas as pd

# import the libraries as shown below

from tensorflow.keras.layers import Input, Lambda, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.applications.inception_v3 import InceptionV3
#from keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble  import GradientBoostingClassifier
from sklearn.svm  import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble  import AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score,f1_score, recall_score, precision_score, auc, roc_auc_score, roc_curve

from sklearn.metrics import classification_report,confusion_matrix
from sklearn.model_selection import cross_val_score
from tensorflow.keras.models import Sequential
import numpy as np
from glob import glob

# Create your views here.

def adminlogout(req):
    messages.info(req,'You are logged out...!')
    return redirect('admin_login')
def admin_index (request):
    return render (request,'admin/index.html')

def admin_pendingusers (request):
        users = UserModels.objects.filter(user_status = 'pending')
        return render(request,'admin/buttons.html', {'users':users})
      

        

def admin_manageusers(request):
    a = UserModels.objects.all()
    paginator = Paginator(a, 5) 
    page_number = request.GET.get('page')
    post = paginator.get_page(page_number)
    return render(request,'admin/dropdowns.html',{'all':post})    

# def admin_upload (req):
#     if req.method == 'POST':
#         file = req.FILES['file']
#         # print(file)
#         file_size = str((file.size)/1024) +' kb'
#         # print(file_size)
#         UploadModels.objects.create(File_size = file_size, Dataset = file)
#         messages.success(req, 'Your dataset was uploaded..')
#     return render(req, 'admin/upload.html')

def admin_upload (req):
    if req.method == 'POST':
        file = req.FILES['file']
        # print(file)
        file_size = str((file.size)/1024) +' kb'
        # print(file_size)
        Upload_dataset_model.objects.create(File_size = file_size, Dataset = file)
        messages.success(req, 'Your dataset was uploaded..')
    return render(req, 'admin/upload.html')

def admin_dataset_btn(request): 
    messages.success(request, 'Dataset uploaded successfully')
    return redirect('admin_upload')
    
def admin_view (req):
    dataset = Upload_dataset_model.objects.all()
    paginator = Paginator(dataset, 5)
    page_number = req.GET.get('page')
    post = paginator.get_page(page_number)
    return render(req, 'admin/view.html', {'data' : dataset, 'user' : post})


def admin_viewdetails(request):
    # df=pd.read_csv('heart.csv')
    data = Upload_dataset_model.objects.last()
    print(data,type(data),'sssss')
    file = str(data.Dataset)
    df = pd.read_csv(f'./media/{file}')
    table = df.to_html(table_id='data_table')
    return render(request,'admin/viewdetails.html', {'t':table})


def delete_dataset(req, id):
    dataset = Upload_dataset_model.objects.get(User_id = id).delete()
    messages.warning(req, 'Dataset was deleted..!')
    return redirect('admin_view')







def admin_graph (request):

   
    details = XG_ALGO.objects.last()
    a = details.Accuracy
    deatails1 = ADA_ALGO.objects.last()
    b = deatails1.Accuracy
    details2 = KNN_ALGO.objects.last()
    c = details2.Accuracy
    deatails3 = SVM_ALGO.objects.last()
    d = deatails3.Accuracy
    details4 = DT_ALGO.objects.last()
    e = details4.Accuracy
    details5 = ANN_ALGO.objects.last()
    f = details5.Accuracy
    details6 = Logistic.objects.last()
    g = details6.Accuracy
    details7 = RandomForest.objects.last()
    h = details7.Accuracy
    return render(request, 'admin/graphanalysis.html', {'xg':a,'ada':b,'knn':c,'sxm':d,'dt':e,'ann':f,'log':g, 'ran':h})

  
  



def admin_reject_btn(req,x):
    user = UserModels.objects.get(user_id = x)
    user.user_status = 'rejected'
    user.save()
    messages.warning(req,'Rejected')
    return redirect('admin_pendingusers')
    
    
def admin_accept_btn(req,x):
    user = UserModels.objects.get(user_id = x)
    user.user_status = 'accepted'
    user.save()
    messages.success(req,'Accepted')
    return redirect('admin_pendingusers')    

def Change_Status(req, id):
    # user_id = req.session['User_Id']
    user = UserModels.objects.get(user_id = id)
    if user.user_status == 'accepted':
        user.user_status = 'rejected'   
        user.save()
        messages.success(req,'Status Changed Succesfully')
        return redirect('admin_manageusers')
    else:
        user.user_status = 'accepted'
        user.save()
        messages.success(req,'Status Changed Successfully')
        return redirect('admin_manageusers')
    
def Delete_User(req, id):
    UserModels.objects.get(user_id = id).delete()
    messages.info(req,'Deleted')
    return redirect('admin_manageusers')


def admin_feedback(request):
    feed = UserFeedbackModels.objects.all()
    return render(request, 'admin/feedback.html',{'back' : feed})


def admin_feedbacksentiments(request):
    feed = UserFeedbackModels.objects.all()
    return render(request,'admin/sentiment.html',{'back' : feed})

def admin_feedebackgraph(request):
    positive = UserFeedbackModels.objects.filter(sentment = 'positive').count()
    very_positive = UserFeedbackModels.objects.filter(sentment = 'very positive').count()
    negative = UserFeedbackModels.objects.filter(sentment = 'negative').count()
    very_negative = UserFeedbackModels.objects.filter(sentment = 'very negative').count()
    neutral = UserFeedbackModels.objects.filter(sentment = 'neutral').count()
    context ={
        'vp': very_positive, 'p':positive, 'n':negative, 'vn':very_negative, 'ne':neutral
    }
    return render(request, 'admin/feedbackgraph.html',context)

 




def admin_svmalgorithm (request):
   return render (request,'admin/svmalgorithm.html')

def admin_logistic (request):
   return render (request,'admin/logisticregression.html')

def admin_desiciontree (request):
   return render (request,'admin/desiciontreealgorithm.html')

def admin_gradientboost (request):
   return render (request,'admin/gradientboostalgorithm.html')
def admin_adaboost (request):
   return render (request,'admin/adaboostalgorithm.html')
def admin_xgboost (request):
   return render (request,'admin/xgboostalgorithm.html')



def admin_forestalgorithm (request): 
   return render (request,'admin/forestalgorithm.html')

def admin_knn (request): 
   return render (request,'admin/knnalgorithm.html')
def KNN_btn(req):
    dataset = Upload_dataset_model.objects.last()
    # print(dataset.Dataset)
    df=pd.read_csv(dataset.Dataset.path)
    X = df.drop('target', axis = 1)
    y = df['target']
    from sklearn.model_selection import train_test_split
    X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=1,test_size=0.2)
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train=scaler.fit_transform(X_train)
    X_test= scaler.transform(X_test)
    from sklearn.neighbors import KNeighborsClassifier
    KNN = KNeighborsClassifier()
    KNN.fit(X_train, y_train)

    # prediction
    train_prediction= KNN.predict(X_train)
    test_prediction= KNN.predict(X_test)
    print('*'*20)

    # evaluation
    from sklearn.metrics import accuracy_score
    accuracy = round(accuracy_score(y_test,test_prediction)*100, 2)
    precession = round(precision_score(y_test,test_prediction,average = 'macro')*100, 2)
    recall = round(recall_score(y_test,test_prediction,average = 'macro')*100, 2)
    f1 = round(f1_score(y_test,test_prediction,average = 'macro')*100, 2)
    # Accuracy_train(accuracy_score(prediction_train,y_train))
    name = "KNN Algorithm"
    KNN_ALGO.objects.create(Accuracy=accuracy,Precession=precession,F1_Score=f1,Recall=recall,Name=name)
    data = KNN_ALGO.objects.last()
    messages.success(req, 'Algorithm executed Successfully')
    return render(req, 'admin/knnalgorithm.html',{'i': data})
    
def logistic_btn(req):
    dataset = Upload_dataset_model.objects.last()
    # print(dataset.Dataset)
    df=pd.read_csv(dataset.Dataset.path)
    X = df.drop('target', axis = 1)
    y = df['target']
    from sklearn.model_selection import train_test_split
    X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=1,test_size=0.2)
    # from sklearn.preprocessing import StandardScaler
    # scaler = StandardScaler()
    # X_train=scaler.fit_transform(X_train)
    # X_test= scaler.transform(X_test)
    from sklearn.linear_model import LogisticRegression
    Logistics = LogisticRegression()
    Logistics.fit(X_train, y_train)

    # prediction
    train_prediction= Logistics.predict(X_train)
    test_prediction= Logistics.predict(X_test)
    print('*'*20)

    # evaluation
    from sklearn.metrics import accuracy_score
    accuracy = round(accuracy_score(y_test,test_prediction)*100, 2)
    precession = round(precision_score(y_test,test_prediction,average = 'macro')*100, 2)
    recall = round(recall_score(y_test,test_prediction,average = 'macro')*100, 2)
    f1 = round(f1_score(y_test,test_prediction,average = 'macro')*100, 2)
    # Accuracy_train(accuracy_score(prediction_train,y_train))
    name = "Logistic Regression Algorithm"
    Logistic.objects.create(Accuracy=accuracy,Precession=precession,F1_Score=f1,Recall=recall,Name=name)
    data = Logistic.objects.last()
    messages.success(req, 'Algorithm executed Successfully')
    return render(req, 'admin/logisticregression.html',{'i': data})
def svm_btn(req):
    dataset = Upload_dataset_model.objects.last()
    # print(dataset.Dataset)
    df=pd.read_csv(dataset.Dataset.path)
    X = df.drop('target', axis = 1)
    y = df['target']
    from sklearn.model_selection import train_test_split
    X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=1,test_size=0.2)
    # from sklearn.preprocessing import StandardScaler
    # scaler = StandardScaler()
    # X_train=scaler.fit_transform(X_train)
    # X_test= scaler.transform(X_test)
    from sklearn.svm import SVC
    svm = SVC()
    svm.fit(X_train, y_train)

    # prediction
    train_prediction= svm.predict(X_train)
    test_prediction= svm.predict(X_test)
    print('*'*20)

    # evaluation
    from sklearn.metrics import accuracy_score
    accuracy = round(accuracy_score(y_test,test_prediction)*100, 2)
    precession = round(precision_score(y_test,test_prediction,average = 'macro')*100, 2)
    recall = round(recall_score(y_test,test_prediction,average = 'macro')*100, 2)
    f1 = round(f1_score(y_test,test_prediction,average = 'macro')*100, 2)
    # Accuracy_train(accuracy_score(prediction_train,y_train))
    name = " SVM Algorithm"
    SVM_ALGO.objects.create(Accuracy=accuracy,Precession=precession,F1_Score=f1,Recall=recall,Name=name)
    data = SVM_ALGO.objects.last()
    messages.success(req, 'Algorithm executed Successfully')
    return render(req, 'admin/svmalgorithm.html',{'i': data})
def decision_btn(req):
    dataset = Upload_dataset_model.objects.last()
    # print(dataset.Dataset)
    df=pd.read_csv(dataset.Dataset.path)
    X = df.drop('target', axis = 1)
    y = df['target']
    from sklearn.model_selection import train_test_split
    X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=1,test_size=0.2)
    # from sklearn.preprocessing import StandardScaler
    # scaler = StandardScaler()
    # X_train=scaler.fit_transform(X_train)
    # X_test= scaler.transform(X_test)
    from sklearn.tree import DecisionTreeClassifier
    decision = DecisionTreeClassifier()
    decision.fit(X_train, y_train)

    # prediction
    train_prediction= decision.predict(X_train)
    test_prediction= decision.predict(X_test)
    print('*'*28)

    # evaluation
    from sklearn.metrics import accuracy_score
    accuracy = round(accuracy_score(y_test,test_prediction)*100, 2)
    precession = round(precision_score(y_test,test_prediction,average = 'macro')*100, 2)
    recall = round(recall_score(y_test,test_prediction,average = 'macro')*100, 2)
    f1 = round(f1_score(y_test,test_prediction,average = 'macro')*100, 2)
    # Accuracy_train(accuracy_score(prediction_train,y_train))
    name = "Decision Tree Algorithm"
    DT_ALGO.objects.create(Accuracy=accuracy,Precession=precession,F1_Score=f1,Recall=recall,Name=name)
    data = DT_ALGO.objects.last()
    messages.success(req, 'Algorithm executed Successfully')
    return render(req, 'admin/desiciontreealgorithm.html',{'i': data})
def random_btn(req):
    dataset = Upload_dataset_model.objects.last()
    # print(dataset.Dataset)
    df=pd.read_csv(dataset.Dataset.path)
    X = df.drop('target', axis = 1)
    y = df['target']
    from sklearn.model_selection import train_test_split
    X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=1,test_size=0.2)
    # from sklearn.preprocessing import StandardScaler
    # scaler = StandardScaler()
    # X_train=scaler.fit_transform(X_train)
    # X_test= scaler.transform(X_test)
    from sklearn.ensemble import RandomForestClassifier
    random = RandomForestClassifier()
    random.fit(X_train, y_train)

    # prediction
    train_prediction= random.predict(X_train)
    test_prediction= random.predict(X_test)
    print('*'*28)

    # evaluation
    from sklearn.metrics import accuracy_score
    accuracy = round(accuracy_score(y_test,test_prediction)*100, 2)
    precession = round(precision_score(y_test,test_prediction,average = 'macro')*100, 2)
    recall = round(recall_score(y_test,test_prediction,average = 'macro')*100, 2)
    f1 = round(f1_score(y_test,test_prediction,average = 'macro')*100, 2)
    # Accuracy_train(accuracy_score(prediction_train,y_train))
    name = "Random Forest Algorithm"
    RandomForest.objects.create(Accuracy=accuracy,Precession=precession,F1_Score=f1,Recall=recall,Name=name)
    data = RandomForest.objects.last()
    messages.success(req, 'Algorithm executed Successfully')
    return render(req, 'admin/forestalgorithm.html',{'i': data})
def gradient_btn(req):
    dataset = Upload_dataset_model.objects.last()
    # print(dataset.Dataset)
    df=pd.read_csv(dataset.Dataset.path)
    X = df.drop('target', axis = 1)
    y = df['target']
    from sklearn.model_selection import train_test_split
    X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=1,test_size=0.2)
    # from sklearn.preprocessing import StandardScaler
    # scaler = StandardScaler()
    # X_train=scaler.fit_transform(X_train)
    # X_test= scaler.transform(X_test)
    from sklearn.ensemble import GradientBoostingClassifier
    gradient = GradientBoostingClassifier()
    gradient.fit(X_train, y_train)

    # prediction
    train_prediction= gradient.predict(X_train)
    test_prediction= gradient.predict(X_test)
    print('*'*28)

    # evaluation
    from sklearn.metrics import accuracy_score
    accuracy = round(accuracy_score(y_test,test_prediction)*100, 2)
    precession = round(precision_score(y_test,test_prediction,average = 'macro')*100, 2)
    recall = round(recall_score(y_test,test_prediction,average = 'macro')*100, 2)
    f1 = round(f1_score(y_test,test_prediction,average = 'macro')*100, 2)
    # Accuracy_train(accuracy_score(prediction_train,y_train))
    name = "Gradient Boost Algorithm"
    ANN_ALGO.objects.create(Accuracy=accuracy,Precession=precession,F1_Score=f1,Recall=recall,Name=name)
    data = ANN_ALGO.objects.last()
    messages.success(req, 'Algorithm executed Successfully')
    return render(req, 'admin/gradientboostalgorithm.html',{'i': data})
def ada_btn(req):
    dataset = Upload_dataset_model.objects.last()
    # print(dataset.Dataset)
    df=pd.read_csv(dataset.Dataset.path)
    X = df.drop('target', axis = 1)
    y = df['target']
    from sklearn.model_selection import train_test_split
    X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=1,test_size=0.2)
   
    from sklearn.ensemble import AdaBoostClassifier
    ada = AdaBoostClassifier()
    ada.fit(X_train, y_train)

    # prediction
    train_prediction= ada.predict(X_train)
    test_prediction= ada.predict(X_test)
    print('*'*28)

    # evaluation
    from sklearn.metrics import accuracy_score
    accuracy = round(accuracy_score(y_test,test_prediction)*100, 2)
    precession = round(precision_score(y_test,test_prediction,average = 'macro')*100, 2)
    recall = round(recall_score(y_test,test_prediction,average = 'macro')*100, 2)
    f1 = round(f1_score(y_test,test_prediction,average = 'macro')*100, 2)
    # Accuracy_train(accuracy_score(prediction_train,y_train))
    name = "ADA Boost Algorithm"
    ADA_ALGO.objects.create(Accuracy=accuracy,Precession=precession,F1_Score=f1,Recall=recall,Name=name)
    data = ADA_ALGO.objects.last()
    messages.success(req, 'Algorithm executed Successfully')
    return render(req, 'admin/adaboostalgorithm.html',{'i': data})
def xg_btn(req):
    dataset = Upload_dataset_model.objects.last()
    # print(dataset.Dataset)
    df=pd.read_csv(dataset.Dataset.path)
    X = df.drop('target', axis = 1)
    y = df['target']
    from sklearn.model_selection import train_test_split
    X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=1,test_size=0.2)
    # from sklearn.preprocessing import StandardScaler
    # scaler = StandardScaler()
    # X_train=scaler.fit_transform(X_train)
    # X_test= scaler.transform(X_test)
    from xgboost import XGBClassifier
    xg = XGBClassifier()
    xg.fit(X_train, y_train)

    # prediction
    train_prediction= xg.predict(X_train)
    test_prediction= xg.predict(X_test)
    print('*'*28)

    # evaluation
    from sklearn.metrics import accuracy_score
    accuracy = round(accuracy_score(y_test,test_prediction)*100, 2)
    precession = round(precision_score(y_test,test_prediction,average = 'macro')*100, 2)
    recall = round(recall_score(y_test,test_prediction,average = 'macro')*100, 2)
    f1 = round(f1_score(y_test,test_prediction,average = 'macro')*100, 2)
    # Accuracy_train(accuracy_score(prediction_train,y_train))
    name = "XG Boost Algorithm"
    XG_ALGO.objects.create(Accuracy=accuracy,Precession=precession,F1_Score=f1,Recall=recall,Name=name)
    data = XG_ALGO.objects.last()
    messages.success(req, 'Algorithm executed Successfully')
    return render(req, 'admin/xgboostalgorithm.html',{'i': data})


