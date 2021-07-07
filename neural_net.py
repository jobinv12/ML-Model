import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix
import streamlit as st


train = pd.read_csv('TrainDataForMotionbtw0-4.csv')
test = pd.read_csv('TestDataForMotionbtw0-4.csv')

#feature and target
X = train.drop('Label',axis=1)
y = train['Label']

#

# Enconding Label column
train['Label'] = 

final_test = test.drop('Label',axis=1)

# train test split
x_train, y_train, x_test, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Neural Net


# scaling 

scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# model

model = Sequential()

model.add(Dense())
model.add(Dropout())

model.compile(optimizer='adam',loss='categorial_crossentropy', metrics=['accuracy'])

earlyStopping = EarlyStopping(monitor='val_loss',mode='min',verbose=1,patience=25)

model.fit(x=x_train,y=y_train, validation_data=(x_test,y_test),epochs=1000, callbacks=[earlyStopping])

# plotting losses

losses = pd.DataFrame(model.history.history)

losses.plot()

# prediction

prediction = model.predict(x_test)

print("classification report")
classification_report(y_test,prediction)

print("confusion matrix")
confusion_matrix(y_test,prediction)

