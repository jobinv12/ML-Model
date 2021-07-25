import numpy as np
import pandas as pd
import streamlit as st
import pickle

models = st.selectbox("Select a model",('SVM Model','Neural Net'))
load_model_btn = st.button("Load Model")

#saved svm model

svm_model_filename = 'svm_model.sav'

# target values
class_name = ["Position_0","Position_1","Position_2","Position_3","Position_4"]
 
st.write("select values for each sensor to predict the position")
col1,col2 = st.beta_columns(2)
col3,col4 = st.beta_columns(2)
col5,col6 = st.beta_columns(2)
col7,col8 = st.beta_columns(2)

#streamlit widgets
with col1:
    sensor0 = st.slider('Sensor0',-1000,1000,0)
with col2:
    sensor1 = st.slider('sensor1',-1000,1000,0)
with col3:
    sensor2 = st.slider('sensor2',-1000,1000,0)
with col4:
    sensor3 = st.slider('sensor3',-1000,1000,0)
with col5:
    sensor4 = st.slider('sensor4',-1000,1000,0)
with col6:
    sensor5 = st.slider('sensor5',-1000,1000,0)
with col7:
    sensor6 = st.slider('sensor6',-1000,1000,0)
with col8:
    sensor7 = st.slider('sensor7',-1000,1000,0)



def svm_model():

    #load model
    load_model = pickle.load(open(svm_model_filename,'rb'))

    result = load_model.predict([[sensor0,sensor1,sensor2,sensor3,sensor4,sensor5,sensor6, sensor7]])

    #convert array to int
    result = np.int(result)

    st.write(class_name[result])


def neural_net_model():
    pass

if load_model_btn:
    if models == 'SVM Model':
        svm_model()
    else:
        st.error("Neural Net model could not be loaded")