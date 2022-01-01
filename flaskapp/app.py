# -*- coding: utf-8 -*-
"""
Created on Tue Dec 21 14:07:29 2021

@author: celine
"""
from flask import Flask,render_template,url_for,request
import pandas as pd 
import pickle
import joblib

from functions_model import *

app = Flask(__name__)

@app.route('/')

def home():
	return render_template('home.html')



@app.route('/predict',methods=['GET','POST'])

def predict():
    
    tags_model = open('tags_model.pkl','rb')
    mlb = open('mlb.pkl','rb')
    tfidf = open('tfidf.pkl', 'rb')
    pca = open('pca.pkl','rb')
    
    clf = joblib.load(tags_model)   
    mlb = joblib.load(mlb) 
    tfidf = joblib.load(tfidf)     
    pca = joblib.load(pca)
    
    if request.method == 'POST':
        message = request.form.get('message') + ' '+ request.form.get('title') 
        text = lemmatize(remove_stop_words(clean_text(message)))
        vect = tfidf.transform([text]).toarray()
        vect = pca.transform(vect)
                
        my_prediction = mlb.inverse_transform(clf.predict(vect))
        
        return render_template('home.html',prediction_text= my_prediction)
    
if __name__ == '__main__':
	app.run(debug=False)


    