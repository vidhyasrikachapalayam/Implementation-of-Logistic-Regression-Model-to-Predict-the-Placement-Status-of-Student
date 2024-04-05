# EX 4 Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1.Import the required packages and print the present data.

2.Print the placement data and salary data.

3.Find the null and duplicate values.

4.Using logistic regression find the predicted values of accuracy , confusion matrices.

5.Display the results.

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by:vidhyasri.k 
RegisterNumber: 212222230170

import pandas as pd
data=pd.read_csv("/content/Placement_Data.csv")
data.head()

data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)
data1.head()

data1.isnull()

data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"])
data1["status"]=le.fit_transform(data1["status"])
data1

x=data1.iloc[:,:-1]
x

y=data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import classification_report
classification_report1=classification_report(y_test,y_pred)
print(classification_report1)

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
*/
```

## Output:
PLACEMENT DATA:

![image](https://github.com/vidhyasrikachapalayam/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119477817/dba30e21-00d1-4cfb-a0a7-c7c3259beb14)

SALARY DATA:

![image](https://github.com/vidhyasrikachapalayam/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119477817/9d297421-9ea7-4ffe-b393-aa7722956da1)

CHECKING THE NULL FUNCTON:

![image](https://github.com/vidhyasrikachapalayam/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119477817/7765a0e5-054a-40f9-b48e-a4df5b137f6f)

DATA DUPLICATE:

![image](https://github.com/vidhyasrikachapalayam/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119477817/b51a0187-6403-4666-ad04-e92ac9c0a654)

PRINT DATA:

![image](https://github.com/vidhyasrikachapalayam/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119477817/7a3743c8-1d67-4216-b570-6bccc3ce733c)

DATA STATUS:

![image](https://github.com/vidhyasrikachapalayam/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119477817/438c947a-9960-40c1-ae3b-c20057fea9a7)

Y PREDICTION ARRAY:

![image](https://github.com/vidhyasrikachapalayam/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119477817/ac59251a-43fa-4163-8c2f-74eae36fc04a)

ACCURACY VALUE:

![image](https://github.com/vidhyasrikachapalayam/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119477817/9ca209e9-df32-42cc-80df-f6a56e464cce)

CLASSIFICATION REPORT:

![image](https://github.com/vidhyasrikachapalayam/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119477817/e744e798-c479-4530-9f6e-9bb1f6542a5a)




## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
