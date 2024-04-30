from django.shortcuts import render , redirect
from userapp.models import *
from adminapp.models import *
from django.contrib import messages
from django.conf import settings
from django.core.mail import send_mail
import urllib.request
import urllib.parse
import random
import string
from django.utils.datastructures import MultiValueDictKeyError
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import numpy as np
import pandas as pd
from django.core.files.storage import default_storage
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score



# model = load_model('model_inception')
# model=load_model(r"C:\Lumpy Disease Full Stack\Lumphy_model.h5")


# Create your views here.

def user_index(request):
    return render(request,'user/index.html')

def user_about(request):
    return render(request,'user/about.html')

def user_contact(request):
    return render(request,'user/contact.html')


def user_services(request):
    return render(request,'user/services.html')

def user_wilddetect(request):
      return render(request,'user/wilddetect.html')

def user_login(request):
    if request.method == 'POST':
        email = request.POST.get('email')
        password = request.POST.get('password')
        print(email,password,'jjjjjjjjjjjjjjjjjj')
        try : 
            user_data = UserModels.objects.get(email = email)
            print(user_data)
             
            if user_data.password ==  password:
                if user_data.user_status == 'accepted':
                    if user_data.Otp_Status == 'verified':
                       messages.success(request,'login successfull')
                       request.session['user_id'] = user_data.user_id
                       print('login sucessfull')
                       return redirect('user_dashboard')
                    else:
                        return redirect('otp')
                elif user_data.password == password and user_data.user_status == 'rejected':
                    messages.warning(request,"your account is rejected")
                else:
                    messages.info(request,"your account is in pending")
            else:
                messages.error(request, 'Error in Email or Password')
        except:
            print('exce[t]')
            return redirect('user_login')
    return render(request,'user/userlogin.html')

def admin_login(request):
    admin_name = 'admin@gmail.com'
    admin_pwd = 'admin'
    if request.method == 'POST':
        a_name = request.POST.get('email')
        a_pwd = request.POST.get('password')
        print(a_name, a_pwd, 'admin entered details')

        if admin_name == a_name and admin_pwd == a_pwd:
            messages.success(request, 'login successful')
            return redirect('admin_index')
        else:
            messages.error(request, 'Wrong Email Or Password')
            return redirect('admin_login')   
    return render(request,'user/adminlogin.html')

def sendSMS(user, otp, mobile):
    data = urllib.parse.urlencode({
        'username': 'Codebook',
        'apikey': '56dbbdc9cea86b276f6c',
        'mobile': mobile,
        'message': f'Hello {user}, your OTP for account activation is {otp}. This message is generated from https://www.codebook.in server. Thank you',
        'senderid': 'CODEBK'
    })
    data = data.encode('utf-8')
    # Disable SSL certificate verification
    # context = ssl._create_unverified_context()
    request = urllib.request.Request("https://smslogin.co/v3/api.php?")
    f = urllib.request.urlopen(request, data)
    return f.read()


def user_registration(request):
    if request.method == 'POST':
        name = request.POST.get('name')
        username = request.POST.get('user')
        email = request.POST.get('email')
        contact = request.POST.get ('contact')
        password = request.POST.get('password')
        file = request.FILES['file']
        # file = request.FILES['file']
        print (request)
        print(name,username,email,contact,password,'data')
        otp = str(random.randint(1000, 9999)) 
        print(otp)
        try:
           print('try')
           UserModels.objects.get(email=email,) 
           messages.info(request, 'Mail already Registered')
           return redirect('user_registration') 
        except:
            print('except')
            # mail message
            mail_message = f'Registration Successfully\n Your 4 digit Pin is below\n {otp}'
            print(mail_message)
            send_mail("Student Password", mail_message, settings.EMAIL_HOST_USER,[email])
            # text nessage
            sendSMS(name, otp, contact)
            UserModels.objects.create( otp=otp,email=email ,password=password,name=name,contact=contact, file=file )        
            request.session['email'] = email
            messages.success(request, 'Register Successfull...!')
            return redirect('user_otp')
    return render(request,'user/register.html')


def user_otp (request):
    user_id = request.session['email']
    user_o =UserModels.objects.get(email = user_id)
    print(user_o,'user available')
    print(type(user_o.otp))
    print(user_o. otp,'created otp')
    # print(user_o. otp, 'creaetd otp')
    if request.method == 'POST':
        u_otp = request.POST.get('otp')
        u_otp = int(u_otp)
        print(u_otp, 'enter otp')
        if u_otp == user_o.otp:
            print('if')
            user_o.Otp_Status  = 'verified'
            user_o.save()
            messages.success(request, 'OTP  verified successfully')
            return redirect('user_login')
        else:
            print('else')
            messages.error(request, 'Error in OTP')
            return redirect('user_otp')
    return render(request,'user/otp.html')

def user_dashboard(request):
    return render(request,'user/dashboard.html')
def user_myprofile(request):
    user_id = request.session['user_id']
    example = UserModels.objects.get(user_id = user_id)
    print(example,'user_id')
    if request.method == 'POST' :
        name = request.POST.get('name')
        email = request.POST.get('email')
        contact = request.POST.get('contact')
        password = request.POST.get('password')
        messages.success(request,'updated successful')

        example.name =name
        example.contact =contact
        example.email =email
        example.password =password
        
        if len(request.FILES)!=0:
            file = request.FILES['file']
            example.file = file
            example.name = name
            example.contact = contact
            example.email = email
            example.password = password
            example.save()
        else:
            example.name = name
            example.email = email
            example.password = password
            example.contact = contact
        #    example.file=file
            example.save()     
    return render(request,'user/myprofile.html',{'i':example})



def user_feedback(request):
    views_id = request.session['user_id']
    user = UserModels.objects.get(user_id = views_id)
    if request.method == 'POST':
        u_feedback = request.POST.get('feedback')
        u_rating = request.POST.get('rating')
        if not user_feedback:
            return redirect('')
        sid=SentimentIntensityAnalyzer()
        score=sid.polarity_scores(u_feedback)
        sentiment=None
        if score['compound']>0 and score['compound']<=0.5:
            sentiment='positive'
        elif score['compound']>=0.5:
            sentiment='very positive'
        elif score['compound']<-0.5:
            sentiment='very negative'
        elif score['compound']<0 and score['compound']>=-0.5:
            sentiment='negative'
        else :
            sentiment='neutral'
        print(sentiment)
        user.star_feedback=u_feedback
        user.star_rating = u_rating
        user.save()
        UserFeedbackModels.objects.create(user_details = user, star_feedback = u_feedback, star_rating = u_rating, sentment= sentiment)
        messages.success(request,'Thankyou For Your Feedback')
    rev=UserFeedbackModels.objects.filter()    
    return render(request,'user/feedback.html')

from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.models import load_model
import numpy as np




from sklearn.ensemble import RandomForestClassifier
def user_detect (req):


# predictdiabetes form Function

    if req.method == 'POST':
        age = req.POST.get('field1')
        sex = req.POST.get('sex')
        cp = req.POST.get('field2')
        trestbps = req.POST.get('field7')
        chol = req.POST.get('field3')
        fbs = req.POST.get('field8')
        restecg	 = req.POST.get('field4')
        thalach  = req.POST.get('field5')
        exang    = req.POST.get('field6')
        oldpeak   = req.POST.get('field9')
        slope	 = req.POST.get('fielda')
        ca       = req.POST.get('fieldb')
        thal	 = req.POST.get('fieldc')
       
        if sex == 0:
            gender = "male"
        else:
            gender = "female"
        context = {'gender': gender}
        
        
       
            
        # print(type(age),x)
        # DATASET.objects.create(Age = age, Glucose = sex, BloodPressure = plasma_CA19_9, SkinThickness = creatinine, Insulin = lyve1, BMI = regb1, DiabetesPedigreeFunction = tff1)
        import pickle

        file_path = 'rfc_hd.pkl'  # Using a raw string
        with open(file_path, 'rb') as file:
            loaded_model = pickle.load(file)

            # res =loaded_model.predict([[age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal ]])
            res =loaded_model.predict([[58,	1,	0,	128	,216.0	,0	,0,	131	,1,	2.2,	1,	3,	3]])
            # res=loaded_model.predict([[25,1,50.125,12.0255,0.15,99.255,45.325]])
            print (res,"yyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy")


            dataset = Upload_dataset_model.objects.last()
            # print(dataset.Dataset)
            df=pd.read_csv(dataset.Dataset.path)

            X = df.drop('target', axis = 1)
            y = df['target']
            # from sklearn.impute import SimpleImputer
            # imputer = SimpleImputer(strategy='mean')
            # X_imputed = imputer.fit_transform(X)
            from sklearn.model_selection import train_test_split
            X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=1,test_size=0.2)
        
            from sklearn.ensemble import RandomForestClassifier
            decision = RandomForestClassifier()
            decision.fit(X_train, y_train)

            # prediction
            train_prediction= decision.predict(X_train)
            test_prediction= decision.predict(X_test)
                    # prediction
            train_prediction= decision.predict(X_train)
            test_prediction= decision.predict(X_test)
            print('*'*20)

            # evaluation
            from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
            accuracy = round(accuracy_score(y_test,test_prediction)*100, 2)
            precession = round(precision_score(y_test,test_prediction,average = 'macro')*100, 2)
            recall = round(recall_score(y_test,test_prediction,average = 'macro')*100, 2)
            f1 = round(f1_score(y_test,test_prediction,average = 'macro')*100, 2)
            x=0
            if res == 0:
                x = 0
                messages.success(req,"Heart Disease Detected")
            else:
                x=1
                messages.warning(req,"Heart Disease Not Detected")
            print(x)
            context = {'accc': accuracy,'pre': precession,'f1':f1,'call':recall,'res':x}
            print(type(res), 'ttttttttttttttttttttttttt')
            print(res)
            
            return render(req, 'user/healthresult.html',context)
    return render (req,'user/coronaryheartdetect.html')


# Result function
def user_detectresult(req):
    return render(req, 'user/healthresult.html')
