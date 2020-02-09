#DINESH SAGAR

#Source Code to find whether a person will leave the organisation or not

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

df=pd.read_csv("hr_analytics.csv")
df.head()

from sklearn.preprocessing import LabelEncoder
ss=LabelEncoder()
df.sales=ss.fit_transform(df.sales)
df.salary=ss.fit_transform(df.salary)
df.head()

x=df.drop(['left'],axis=1)
y=df.left

from sklearn.preprocessing import MinMaxScaler
mm=MinMaxScaler()
x=mm.fit_transform(x)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=1,test_size=0.2)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(C=100,solver='lbfgs')
lr.fit(x_train,y_train)

predicition=lr.predict(x_test)

from sklearn.metrics import accuracy_score, confusion_matrix
ds=accuracy_score(predicition,y_test)
print(ds)

ds1=confusion_matrix(y_test,predicition)
print(ds1)

from sklearn.svm import SVC
sv=SVC(C=100)
sv.fit(x_train,y_train)
print(sv.score(x_test,y_test))

