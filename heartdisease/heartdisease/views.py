from django.shortcuts import render
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def home(request):
    return render(request, 'home.html')
def prediction(request):
    return render(request, 'prediction.html')
def about(request):
    return render(request, 'about.html')
def contact(request):
    return render(request, 'contact.html')
def team(request):
    return render(request, 'team.html')
def result(request):
    df = pd.read_csv(r'C:\Users\user\Documents\Applications\INTERNSHIP\NITDA_Internship\Naidt\HD_Project\Cleaned_HD_Dataset.csv')

    x = df.drop(['target'], axis=1)
    y = df['target']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    model = LogisticRegression()
    model.fit(x_train, y_train)

    val1 = float(request.GET['name1'])
    val2 = float(request.GET['name2'])
    val3 = float(request.GET['name3'])
    val4 = float(request.GET['name4'])
    val5 = float(request.GET['name5'])
    val6 = float(request.GET['name6'])
    val7 = float(request.GET['name7'])
    val8 = float(request.GET['name8'])
    val9 = float(request.GET['name9'])
    

    pred = model.predict([[val1, val2, val3, val4, val5, val6, val7, val8, val9]])

    result1=""
    if pred == [1]:
        result1 = 'Positive!! You Have A Heart Condition'
    else:
        result1 = 'Negative!! Your Heart Looks Good'
        
    return render(request, 'prediction.html', {'result2':result1})