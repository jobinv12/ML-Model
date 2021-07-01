import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.preprocessing import MinMaxScaler,LabelEncoder
from sklearn.svm import SVC
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix



#loading Dataset
df = pd.read_csv("Motion_sensor_data2021-Jun-25-14-14-43.csv")

#feature and target

X = []
y = []

# train test split
x_train, y_train, x_test, y_test = train_test_split(X,y,  test_size=0.3, random_state=42)

# Sklearn SVM

svm = SVC()
svm.fit(x_train,y_train)
prdeict_svm = svm.predict(x_test)


print("confusion matrix")
confusion_matrix(y_test,prdeict_svm)











# Neural Net

# Enconding Label column

# le = LabelEncoder()

# df['Label'] = le.fit_transform(df['Label'])

# train test split

# X = df.drop('Label',axis=1).values
# y = df['Label'].values

# x_train,y_train,x_test,y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# scaling 

# scaler = MinMaxScaler()

# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)

# model

# model = Sequential()

# model.add(Dense())
# model.add(Dropout())

# model.compile(optimizer='rmsprop',loss='categorial_crossentropy', metrics=['accuracy'])

# earlyStopping = EarlyStopping(monitor='val_loss',mode='min',verbose=1,patience=25)

# model.fit(x=x_train,y=y_train, validation_data=(x_test,y_test),epochs=1000, callbacks=[earlyStopping])

# plotting losses

# losses = pd.DataFrame(model.history.history)

# losses.plot()

#prediction

# prediction = model.predict(x_test)

# print("classification report")
# classification_report(y_test,prediction)

# print("confusion matrix")
# confusion_matrix(y_test,prediction)