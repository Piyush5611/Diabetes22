import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
data=pd.read_csv('F:\SONGS\DataSets-master (1)\DataSets-master\diabetes.csv')
data.head(10)
sns.countplot(x='Outcome',data=data)
data['Age'].plot.hist()
data.info()
zero=['Glucose','BloodPressure','SkinThickness','BMI','Insulin']
for column in zero:
    data[column]= data[column].replace(0,np.NaN)
data.isnull()
data.isnull().sum()
sns.heatmap(data.isnull(),yticklabels=False,cmap='viridis')
for column in zero:
    mean=int(data[column].mean(skipna=True))
    data[column]=data[column].replace(np.NaN,mean)
real_x=data[['Glucose','BloodPressure','Insulin','BMI','Age']]
real_y=data.iloc[:,8].values    
from sklearn.model_selection import train_test_split
train_x,test_x,train_y,test_y=train_test_split(real_x,real_y,test_size=0.2,random_state=0)
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
train_x=sc.fit_transform(train_x)
test_x=sc.fit_transform(test_x)
from sklearn.neighbors import KNeighborsClassifier
kns=KNeighborsClassifier(n_neighbors=11,metric='euclidean',p=2)
kns.fit(train_x,train_y)
pred=kns.predict(test_x)
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(test_y,pred)
cm

pickle.dump(kns,open('model.pkl','wb'))
