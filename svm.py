import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.preprocessing import MinMaxScaler,LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import streamlit as st
import pickle

train = pd.read_csv('TrainDataForMotionbtw0-4.csv)

#feature and target
X = train.drop('Label',axis=1)
y = train['Label']

# train test split
x_train, y_train, x_test, y_test = train_test_split(X,y,  test_size=0.3, random_state=42)
# Sklearn SVM

def svm_model():

    grid_params = {'C':[0.1,1,10,100,1000],'gamma':[1,0.1,0.01,0.001,0.0001],'kernel':['rbf']}

    grid_svm = GridSearchCV(SVC(),grid_params, verbose=1)
    grid_svm.fit(x_train,y_train)
    st.write(grid_svm.best_params_)
    predict_svm = grid_svm.predict(x_test)
    
    st.write("confusion matrix")
    confusion_matrix(y_test,prdeict_svm)
    st.write("Classification Report")
    classification_report(y_test,prdeict_svm)