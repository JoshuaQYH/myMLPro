import pandas as pd 
import csv
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from LinearRegression import linearRegression
from sklearn import preprocessing 
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
"""
Task: 
1. Read the dataset and extract the features and label. 
2. Train the linear regression model and predict test data.
"""

# get the origin train data.
ori_train_data = pd.read_csv("./train.csv", encoding='Big5')
# print(ori_train_data.head())
ori_test_data = pd.read_csv("./test.csv", encoding= 'Big5')


# get the attibutes' value
feature1 = ['0','1','2','3','4','5','6','7','8','9',
'10','11','12','13','14','15','16','17','18','19','20','21','22','23']
second_train_data = ori_train_data[feature1] 
#print(second_train_data.head())
feature2 = ['0','1','2','3','4','5','6','7','8']
second_test_data = ori_test_data[feature2]


# get the transpose object and make every row data to be obersevation data at the same time.
data_format = {"AMB_TEMP":[], "CH4":[], "CO":[], "NMHC":[],"NO":[], "NO2":[],"NOx":[], "O3":[], "PM10":[], "PM2.5":[], "RAINFALL":[],"RH":[],"SO2":[],"THC":[],"WD_HR":[],"WIND_DIREC":[],"WIND_SPEED":[],"WS_HR":[]} 
train_data = pd.DataFrame(data_format)

for i in range(int(second_train_data.index.size)):
    if i % 18 == 0:
        tmp_df = second_train_data.iloc[i:i+18, :].T
        tmp_df.columns = ['AMB_TEMP','CH4','CO','NMHC',"NO","NO2","NOx","O3","PM10","PM2.5","RAINFALL","RH","SO2","THC","WD_HR","WIND_DIREC","WIND_SPEED", "WS_HR"]
        itera_time = int(i / 18)
        #print(tmp_df)
        tmp_df.index = [j for j in range(itera_time * 24 + 24) if j >= itera_time*24]
        train_data = pd.concat([train_data, tmp_df])
#print(train_data)
# The train_data here is a 5760 rows * 18 columns matrix. As the num of row increases, the time increases.

test_data = pd.DataFrame(data_format)
#print(second_test_data)
for i in range(int(second_test_data.index.size)):
    if i % 18 == 0:
        tmp_df = second_test_data.iloc[i:i+18, :].T
        tmp_df.columns = ['AMB_TEMP','CH4','CO','NMHC',"NO","NO2","NOx","O3","PM10","PM2.5","RAINFALL","RH","SO2","THC","WD_HR","WIND_DIREC","WIND_SPEED", "WS_HR"]
        itera_time = int(i / 18)
        #print(tmp_df)
        tmp_df.index = [j for j in range(itera_time * 9 + 9) if j >= itera_time*9]
        test_data = pd.concat([test_data, tmp_df])
#print(train_data)
# The train_data here is a 5760 rows * 18 columns matrix. As the num of row increases, the time increases.


def delete_NR(x):
    if(x == "NR"):
        return 0
    return x

# data clean
train_data["RAINFALL"] = train_data["RAINFALL"].apply(delete_NR)
test_data["RAINFALL"] = test_data["RAINFALL"].apply(delete_NR)
#print(train_data.info())
train_data = train_data.apply(lambda x:x.astype(float))
test_data = test_data.apply(lambda x:x.astype(float))


# visualize 
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
train_data_corr = train_data.corr()
plt.figure(figsize = (14,14))
#print(train_data_corr)
sns.heatmap(train_data_corr, annot = True)
#plt.show()


# extract the features and label.
# The strategy of extracting is to select the 1~10th hours as the final 
# train data and the 11st hour's PM2.5 as the label
train_label_y = []
train_data_X = []


for i in range(train_data.index.size - 9):
    tmp_list = train_data[i : i + 9].values.reshape(1, train_data.columns.size * 9)[0].tolist()
    tmp_list.append(1)
    train_data_X.append(tmp_list)
    train_label_y.append(train_data.iat[i + 9, 9])
#print(train_data_X)
train_data_X = np.array(train_data_X)
train_label_y = np.array(train_label_y)

test_label_y = []
test_data_X = []
for i in range(test_data.index.size - 9):	
    tmp_list = test_data[i : i + 9].values.reshape(1, test_data.columns.size * 9)[0].tolist()
    tmp_list.append(1)
    test_data_X.append(tmp_list) 
print("---------------\n")
test_data_X = np.array(test_data_X)
test_label_y = np.array(test_label_y)


# scale the train data
mm = preprocessing.MinMaxScaler()
train_data_X = mm.fit_transform(train_data_X)
test_data_X = mm.fit_transform(test_data_X)


# train the model
linear_classifer = linearRegression(train_data.columns.size * 9, 1000, 1)
#print(train_data_X)
#print(train_label_y)

x_train, x_test, y_train, y_test = train_test_split(train_data_X, train_label_y, test_size = 0.3)

# use the sklearn's linear regression 
classifier = LinearRegression()
classifier.fit(x_train, y_train)
predict_y = classifier.predict(x_test)

# predict the test data and print out score
result_file = open("result.csv", 'w')
result_file.write("id,PM2.5,\n")
for i in range(len(test_data_X)):
    result_file.write(str(i))
    result_file.write(',')
    result_file.write(str(classifier.predict(test_data_X[i].reshape(1,-1))[0]))
    result_file.write('\n')
result_file.close()

# Here are the model by myself.
"""
print("start to train!")
linear_classifer.fit(x_train, y_train)
print('train done!')
rate = linear_classifer.score(x_test, y_test)
print('accuracy rate: %.4lf' % rate)

# predict the test data and print out score
result_file = open("result.csv", 'w')
result_file.write("id,PM2.5,\n")
for i in range(len(test_data_X)):
    result_file.write(str(i))
    result_file.write(',')
    result_file.write(str(linear_classifer.predict(test_data_X[i])))
    result_file.write('\n')
result_file.close()
"""






