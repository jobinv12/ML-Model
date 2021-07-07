import pandas as pd
import numpy as np
import svm
import neural_net


model_option = st.selectbox('Select a ML Model you want to run',('select one','SVM','Neural Net'))

progress_bar = st.progress(0)
status = st.empty()

#empty dataframe
full_data = pd.DataFrame()

#loading Dataset
data = ['Motion_sensor_data2021-Jul-01-22-47-30','Motion_sensor_data2021-Jul-01-22-47-31','Motion_sensor_data2021-Jul-01-22-47-32','Motion_sensor_data2021-Jul-01-22-47-33','Motion_sensor_data2021-Jul-01-22-47-34','Motion_sensor_data2021-Jul-01-22-47-35','Motion_sensor_data2021-Jul-01-22-47-36','Motion_sensor_data2021-Jul-01-22-47-37']

for dataset in data:
    df = pd.read_csv("{}.csv".format(dataset))
    full_data =full_data.append(df)

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