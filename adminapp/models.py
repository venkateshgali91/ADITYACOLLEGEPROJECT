from django.db import models

# Create your models here.

# ANM Algorithm

class Upload_dataset_model(models.Model):
    User_id = models.AutoField(primary_key = True)
    Dataset = models.FileField(null=True)
    File_size = models.CharField(max_length = 100) 
    Date_Time = models.DateTimeField(auto_now = True)
    
    class Meta:
        db_table = 'upload_table'

class ANM_ALGO(models.Model):
    ANM_ID = models.AutoField(primary_key = True)
    Accuracy = models.TextField(max_length = 100)
    Precession = models.TextField(max_length = 100) 
    F1_Score = models.TextField(max_length = 100)
    Recall = models.TextField(max_length = 100)
    Name = models.TextField(max_length = 100)
    
    class Meta:
        db_table = 'ANM_algo'
        
# Logistic Regression
class ANN_ALGO(models.Model):
    ANN_ID = models.AutoField(primary_key = True)
    Accuracy = models.TextField(max_length = 100)
    Precession = models.TextField(max_length = 100) 
    F1_Score = models.TextField(max_length = 100)
    Recall = models.TextField(max_length = 100)
    Name = models.TextField(max_length = 100)
    
    class Meta:
        db_table = 'ANN_ALGO'

class Logistic(models.Model):
    Logistic_ID = models.AutoField(primary_key = True)
    Accuracy = models.TextField(max_length = 100)
    Precession = models.TextField(max_length = 100) 
    F1_Score = models.TextField(max_length = 100)
    Recall = models.TextField(max_length = 100)
    Name = models.TextField(max_length = 100)
    
    class Meta:
        db_table = 'Logistic'
        
class RandomForest(models.Model):
    Random_ID = models.AutoField(primary_key = True)
    Accuracy = models.TextField(max_length = 100)
    Precession = models.TextField(max_length = 100) 
    F1_Score = models.TextField(max_length = 100)
    Recall = models.TextField(max_length = 100)
    Name = models.TextField(max_length = 100)
    
    class Meta:
        db_table = 'RandomForest'


# XG-Boost Algo
class XG_ALGO(models.Model):
    XG_ID = models.AutoField(primary_key = True)
    Accuracy = models.TextField(max_length = 100)
    Precession = models.TextField(max_length = 100) 
    F1_Score = models.TextField(max_length = 100)
    Recall = models.TextField(max_length = 100)
    Name = models.TextField(max_length = 100)
    
    class Meta:
        db_table = 'XG_algo'

# ADA Boost Algo
class ADA_ALGO(models.Model):
    XG_ID = models.AutoField(primary_key = True)
    Accuracy = models.TextField(max_length = 100)
    Precession = models.TextField(max_length = 100) 
    F1_Score = models.TextField(max_length = 100)
    Recall = models.TextField(max_length = 100)
    Name = models.TextField(max_length = 100)
    
    class Meta:

        db_table = 'ADA_algo'

# KNN Algo
class KNN_ALGO(models.Model):
    XG_ID = models.AutoField(primary_key = True)
    Accuracy = models.TextField(max_length = 100)
    Precession = models.TextField(max_length = 100) 
    F1_Score = models.TextField(max_length = 100)
    Recall = models.TextField(max_length = 100)
    Name = models.TextField(max_length = 100)
    
    class Meta:
        db_table = 'KNN_algo'

# SXM Algo
class SVM_ALGO(models.Model):
    SXM_ID = models.AutoField(primary_key = True)
    Accuracy = models.TextField(max_length = 100)
    Precession = models.TextField(max_length = 100) 
    F1_Score = models.TextField(max_length = 100)
    Recall = models.TextField(max_length = 100)
    Name = models.TextField(max_length = 100)
    
    class Meta:
        db_table = 'SXM_algo'

# DECISION TREE Algo
class DT_ALGO(models.Model):
    DT_ID = models.AutoField(primary_key = True)
    Accuracy = models.TextField(max_length = 100)
    Precession = models.TextField(max_length = 100) 
    F1_Score = models.TextField(max_length = 100)
    Recall = models.TextField(max_length = 100)
    Name = models.TextField(max_length = 100)
    
    class Meta:
        db_table = 'DT_algo'

# dataset
class DATASET(models.Model):
    DS_ID = models.AutoField(primary_key = True)
    Age = models.IntegerField()
    Glucose = models.IntegerField() 
    BloodPressure = models.IntegerField()
    SkinThickness = models.IntegerField()
    Insulin = models.IntegerField()
    BMI = models.IntegerField()
    DiabetesPedigreeFunction = models.IntegerField() 
    Pregnancies = models.IntegerField()

    
    class Meta:
        db_table = 'Dataset'
