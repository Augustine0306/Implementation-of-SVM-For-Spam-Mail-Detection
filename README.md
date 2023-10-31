# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1.Import the necessary packages.

2.Read the given csv file and display the few contents of the data.

3.Assign the features for x and y respectively.

4.Split the x and y sets into train and test sets.

5.Convert the Alphabetical data to numeric using CountVectorizer.

6.Predict the number of spam in the data using SVC (C-Support Vector Classification) method of SVM (Support vector machine) in sklearn library.

7.Find the accuracy of the model.

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: AUGUSTINE J
RegisterNumber:  212222240015
*/
```
```python
import chardet
file='/content/spam (1).csv'
with open(file, 'rb') as rawdata:
     print('Result output')
    result = chardet.detect(rawdata.read(10000))
result

import pandas as pd
data=pd.read_csv("/content/spam (1).csv",encoding="windows-1252")

data.head()

data.info()

data.isnull().sum()

x=data["v1"].values

y=data["v2"].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)


from sklearn.feature_extraction.text import CountVectorizer 
cv=CountVectorizer()

x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)

from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)

y_pred=svc.predict(x_test)
print("y_pred")
y_pred


from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
print("accuracy")
accuracy
```

## Output:
![image](https://github.com/Augustine0306/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119404460/2617a019-3912-45d5-a7b8-1e0b1444ef10)

![image](https://github.com/Augustine0306/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119404460/9b93b9c7-09ae-429e-ae47-7fd7b8b71bd3)

![image](https://github.com/Augustine0306/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119404460/6cea5a3d-36a4-4c12-a416-99ef28c37cff)

![image](https://github.com/Augustine0306/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119404460/0a679a26-02b3-4a26-94b6-dfec8369ca7f)

![image](https://github.com/Augustine0306/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119404460/706f13ec-a1fe-4c0c-a37f-69b77f05e5f7)

![image](https://github.com/Augustine0306/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119404460/f981e1f0-f875-4465-b9f2-1ff73fd0707f)


## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
