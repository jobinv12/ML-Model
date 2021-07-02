# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split,GridSearchCV
# from sklearn.preprocessing import MinMaxScaler,LabelEncoder
# from sklearn.svm import SVC
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense,Dropout,
# from tensorflow.keras.callbacks import EarlyStopping
# from sklearn.metrics import classification_report, confusion_matrix

# model_option = st.selectbox('Select a ML Model you want to run',('SVM','Neural Net'))

# #loading Dataset
# full_data
# dataset_list = ["Motion_sensor_data2021-Jul-01-22-47-30",'Motion_sensor_data2021-Jul-01-22-47-31.csv','Motion_sensor_data2021-Jul-01-22-47-32.csv','Motion_sensor_data2021-Jul-01-22-47-33.csv','Motion_sensor_data2021-Jul-01-22-47-34.csv','Motion_sensor_data2021-Jul-01-22-47-35.csv','Motion_sensor_data2021-Jul-01-22-47-36.csv','Motion_sensor_data2021-Jul-01-22-47-37.csv']

# for dataset in dataset_list:
#     data = pd.read_csv("{}.csv".format(dataset))
#     full_data.join(data,how='full')

# full_data.head
# #feature and target

# X = []
# y = []

# # train test split
# x_train, y_train, x_test, y_test = train_test_split(X,y,  test_size=0.3, random_state=42)

# # Sklearn SVM

# def svm_model(x_train,y_train,x_test,y_test):

#     grid_params = {'C':[0.1,1,10,100,1000],'gamma':[1,0.1,0.01,0.001,0.0001],'kernel':['rbf']}

#     grid_svm = GridSearchCV(SVC(),grid_params, verbose=1)
#     grid_svm.fit(x_train,y_train)
#     st.write(grid_svm.best_params_)
#     predict_svm = grid_svm.predict(x_test)
    
#     st.write("confusion matrix")
#     confusion_matrix(y_test,prdeict_svm)
#     st.write("Classification Report")
#     classification_report(y_test,prdeict_svm)



# svm_model(x_train,y+train,x_test,y_test)





# # Neural Net

# # Enconding Label column

# # le = LabelEncoder()

# # df['Label'] = le.fit_transform(df['Label'])

# # train test split

# # X = df.drop('Label',axis=1).values
# # y = df['Label'].values

# # x_train,y_train,x_test,y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# # scaling 

# # scaler = MinMaxScaler()

# # X_train = scaler.fit_transform(X_train)
# # X_test = scaler.transform(X_test)

# # model

# # model = Sequential()

# # model.add(Dense())
# # model.add(Dropout())

# # model.compile(optimizer='adam',loss='categorial_crossentropy', metrics=['accuracy'])

# # earlyStopping = EarlyStopping(monitor='val_loss',mode='min',verbose=1,patience=25)

# # model.fit(x=x_train,y=y_train, validation_data=(x_test,y_test),epochs=1000, callbacks=[earlyStopping])

# # plotting losses

# # losses = pd.DataFrame(model.history.history)

# # losses.plot()

# #prediction

# # prediction = model.predict(x_test)

# # print("classification report")
# # classification_report(y_test,prediction)

# # print("confusion matrix")
# # confusion_matrix(y_test,prediction)