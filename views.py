from django.db.models import Count
from django.db.models import Q
from django.shortcuts import render, redirect, get_object_or_404
import datetime
import openpyxl

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import re
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import VotingClassifier

# Create your views here.
from Remote_User.models import ClientRegister_Model,tweets_prediction,detection_ratio,detection_accuracy

def login(request):


    if request.method == "POST" and 'submit1' in request.POST:

        username = request.POST.get('username')
        password = request.POST.get('password')
        try:
            enter = ClientRegister_Model.objects.get(username=username,password=password)
            request.session["userid"] = enter.id

            return redirect('ViewYourProfile')
        except:
            pass

    return render(request,'RUser/login.html')

def Register1(request):

    if request.method == "POST":
        username = request.POST.get('username')
        email = request.POST.get('email')
        password = request.POST.get('password')
        phoneno = request.POST.get('phoneno')
        country = request.POST.get('country')
        state = request.POST.get('state')
        city = request.POST.get('city')
        ClientRegister_Model.objects.create(username=username, email=email, password=password, phoneno=phoneno,
                                            country=country, state=state, city=city)

        return render(request, 'RUser/Register1.html')
    else:
        return render(request,'RUser/Register1.html')

def ViewYourProfile(request):
    userid = request.session['userid']
    obj = ClientRegister_Model.objects.get(id= userid)
    return render(request,'RUser/ViewYourProfile.html',{'object':obj})


def Tweet_Account_Prediction(request):
    if request.method == "POST":
        kword = request.POST.get('keyword')
        if request.method == "POST":
            news = request.POST.get('keyword')

        df = pd.read_csv("Tweets_DataSets.csv")
        Tweet = []
        Labels = []
        df['label'] = df['account_type'].map({'human': 0, 'bot': 1})
        df['message'] = df['description']
        df.drop(['account_type', 'description'], axis=1, inplace=True)
        X = df['message']
        y = df['label']

        from sklearn.feature_extraction.text import CountVectorizer

        cv = CountVectorizer(lowercase=False, strip_accents='unicode', ngram_range=(1, 1))
        X = cv.fit_transform(X.values.astype('U'))  # Fit the Data .values.astype('U')

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

        predictors = []

        print("SVM")
        # SVM Model
        from sklearn import svm

        lin_clf = svm.LinearSVC()
        lin_clf.fit(X_train, y_train)
        predict_svm = lin_clf.predict(X_test)

        # Logistic Regression Model
        print("Logistic Regression")
        from sklearn.linear_model import LogisticRegression
        logreg = LogisticRegression(random_state=42)
        logreg.fit(X_train, y_train)
        predict_log = logreg.predict(X_test)

        print("Naive Bayes")

        from sklearn.naive_bayes import MultinomialNB
        NB = MultinomialNB()
        NB.fit(X_train, y_train)
        predict_nb = NB.predict(X_test)

        data = [kword]
        vect = cv.transform(data).toarray()
        my_prediction = NB.predict(vect)

        if my_prediction == 1:
            val = 'Bot'

        else:
            val = 'Normal'

        print(val)

        tweets_prediction.objects.create(Tweet_Message=news,Prediction=val)

        return render(request, 'RUser/Tweet_Account_Prediction.html',{'objs': val})
    return render(request, 'RUser/Tweet_Account_Prediction.html')



