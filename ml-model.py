import pandas as pd
import numpy as np
import svm
import neural_net


model_option = st.selectbox('Select a ML Model you want to run',('select one','SVM','Neural Net'))

progress_bar = st.progress(0)
status = st.empty()

# train and test data
train = pd.read_csv('TrainDataForMotionbtw0-4.csv')
test = pd.read_csv('TestDataForMotionbtw0-4.csv')

#feature and target
X = train.drop('Label',axis=1)
y = train['Label']

final_test = test.drop('Label',axis=1)

# train test split
x_train, y_train, x_test, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# if model_option == "SVM":
#     progress_bar.progress(0)
#     status.empty()
#     # svm_model(x_train,y_train,x_test,y_test)
# elif model_option == "Neural Net":
#     progress_bar.progress(0)
#     status.empty()
#     st.error("Neural Net Model Build Failed.")
# else:
#     st.error("Select a model")