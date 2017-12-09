from sklearn.externals.joblib import Memory
from sklearn.datasets import load_svmlight_file

def get_data():
    data = load_svmlight_file("housing_scale.txt")
    return data[0], data[1]

X, y = get_data()	

import numpy as np
from sklearn.model_selection import train_test_split

X = X.dot(np.eye(X.shape[1]))
o = np.ones((X.shape[0],1))
X = np.hstack((X,o))		
y = y.reshape((y.shape[0],1))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
w = np.zeros((13,1))

n = 100 	

Ltrain = np.zeros((n))	

Ltest = np.zeros((n))	

def Lfun(w,X,y):	
    m = y.shape[0]
    o = np.ones((m,1))
    l = o-(X.dot(w))*y
    for i in range (m):
        if l[i] < 0:
            l[i] = 0
    return l.sum()+0.5*np.sum(w*w)

def DER(w,X,y):		
    m = y.shape[0]
    j = (X.dot(w))*y
    o = np.zeros((m,1))
    for i in range (m):
        if j[i] < 1:
            o[i] = y[i]
    return -((X.T).dot(o))

for i in range (n):	
   
    G = (DER(w,X_train,y_train))
    w = w - 0.1*G
    Ltrain[i] = Lfun(w,X_train,y_train)
    Ltest[i] = Lfun(w,X_test,y_test)

import matplotlib.pyplot as plt		
x = np.arange(0,n,1)
plt.plot(x, Ltrain,  'r',label = 'training')
plt.plot(x, Ltest,  'b',label = 'testing')
plt.legend(loc='upper right')
plt.xlabel('Times of iteration')
plt.ylabel('Loss')
plt.show()