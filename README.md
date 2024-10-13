# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required packages and print the present data.
2. Print the placement data and salary data.
3. Find the null and duplicate values.
4. Using logistic regression find the predicted values of accuracy , confusion matrices.
5. Finally execute the program and display the output.

## Program and Output :
```
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: ALIYA SHEEMA
RegisterNumber:  212223230011


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
dataset=pd.read_csv("/content/Placement_Data_Full_Class (1).csv")
dataset.head()
```

![image](https://github.com/user-attachments/assets/7277661a-586a-4ed7-9b64-a82b14d06893)

```
dataset.tail()
```

![image](https://github.com/user-attachments/assets/3c9ccf92-3c27-4988-9161-73790f57303d)

```
dataset.info()
```

![image](https://github.com/user-attachments/assets/110c98ae-1ae0-4060-b9dd-83a59a3aca99)

```
dataset=dataset.drop('sl_no',axis=1)
dataset["gender"]=dataset["gender"].astype('category')
dataset["ssc_b"]=dataset["ssc_b"].astype('category')
dataset["hsc_b"]=dataset["hsc_b"].astype('category')
dataset["degree_t"]=dataset["degree_t"].astype('category')
dataset["workex"]=dataset["workex"].astype('category')
dataset["specialisation"]=dataset["specialisation"].astype('category')
dataset["status"]=dataset["status"].astype('category')
dataset["hsc_s"]=dataset["hsc_s"].astype('category')
dataset.dtypes
```

![image](https://github.com/user-attachments/assets/12ef78bf-cb81-49ba-bd92-550a15e20b60)

```
dataset["gender"]=dataset["gender"].cat.codes
dataset["ssc_b"]=dataset["ssc_b"].cat.codes
dataset["hsc_b"]=dataset["hsc_b"].cat.codes
dataset["degree_t"]=dataset["degree_t"].cat.codes
dataset["workex"]=dataset["workex"].cat.codes
dataset["specialisation"]=dataset["specialisation"].cat.codes
dataset["status"]=dataset["status"].cat.codes
dataset["hsc_s"]=dataset["hsc_s"].cat.codes
dataset
```

![image](https://github.com/user-attachments/assets/c3031c7b-8e77-44f1-ad87-a7a6d0a5cefa)

```
dataset.info()
```

![image](https://github.com/user-attachments/assets/84dfc75d-905c-4dbb-87b6-7f34782feeaa)

```
X=dataset.iloc[:,:-1].values
Y=dataset.iloc[:,-1].values
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=1)
clf=LogisticRegression()
clf.fit(x_train,y_train)
```

![image](https://github.com/user-attachments/assets/cb4f837d-fd8a-4ffb-9bb1-3416fc7afc48)

```
y_pred=clf.predict(x_test)
clf.score(x_test,y_test)
```

![image](https://github.com/user-attachments/assets/a1f749a3-a11c-4584-bdcc-c780e78ea191)

```
from sklearn.metrics import  accuracy_score, confusion_matrix
cf=confusion_matrix(y_test, y_pred)
cf
```

![image](https://github.com/user-attachments/assets/06290a0d-da79-4b7c-bf2d-fb7a3634e858)

```
accuracy=accuracy_score(y_test, y_pred)
accuracy
```

![image](https://github.com/user-attachments/assets/6e4bac3b-bcdf-4e73-8cd4-bfe49fb022ac)

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
