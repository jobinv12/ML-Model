import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.preprocessing import MinMaxScaler,LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import streamlit as st
import pickle

train = pd.read_csv('TrainDataForMotionbtw0-4.csv')
test = pd.read_csv('TestDataForMotionbtw0-4.csv')

#matrix multiply all columns with 1
# train['Sensor0'] = np.dot(train['Sensor0'],[1]).reshape(3,1)
# # train['Sensor1'] = np.dot(train['Sensor1'],[1])
# # train['Sensor2'] = np.dot(train['Sensor2'],[1])
# # train['Sensor3'] = np.dot(train['Sensor3'],[1])
# # train['Sensor4'] = np.dot(train['Sensor4'],[1])
# # train['Sensor5'] = np.dot(train['Sensor5'],[1])
# # train['Sensor6'] = np.dot(train['Sensor6'],[1])
# # train['Sensor7'] = np.dot(train['Sensor7'],[1])

column_names = ['Sensor0','Sensor1','Sensor2','Sensor3','Sensor4','Sensor5','Sensor6','Sensor7']


for i in range(0,7):
    train[column_names[i]] = np.dot(train[column_names[i]],[1])
    train[column_names[i]].reshape(1,1)

st.write(train.head())

#feature and target
X = train.drop('Label',axis=1)
y = train['Label']

# label encode Target
le = LabelEncoder()
train['Label'] = le.fit_transform(train['Label'])



# Resting postions 
# Position_0,Position_1,Position_2,Position_3,Position_4,Resting_Position
# class_name = ["Position_0","Position_1","Position_2","Position_3","Position_4","Resting_Position"]

final_test = test.drop('Label',axis=1)

# train test split
x_train, y_train, x_test, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# SVM

# def svm_model():

#     grid_params = {'C':[0.1,1,10,100,1000],'gamma':[1,0.1,0.01,0.001,0.0001],'kernel':['rbf']}

#     grid_svm = GridSearchCV(SVC(),grid_params, verbose=1)
#     grid_svm.fit(x_train,y_train)
#     st.write(grid_svm.best_params_)
#     predict_svm = grid_svm.predict(x_test)
    
#     st.write("confusion matrix")
#     confusion_matrix(y_test,prdeict_svm)
#     st.write("Classification Report")
#     classification_report(y_test,prdeict_svm)


# svm_model()