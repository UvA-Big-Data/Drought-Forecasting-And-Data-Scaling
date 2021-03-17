import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from sklearn import utils

pd.set_option('display.max_columns', 100)

# Limit size so it is still managable, experiment with these nrows values to see what is possible
train_set = pd.read_csv('train_timeseries/train_timeseries.csv', nrows=10000)
test_set = pd.read_csv('train_timeseries/train_timeseries.csv', nrows=2000)

train_set.dropna(inplace=True)
test_set.dropna(inplace=True)

train_set.drop(columns=['date', 'fips'], inplace=True)
test_set.drop(columns=['date', 'fips'], inplace=True)

y_train = train_set.iloc[:,-1].astype(int) # as integer else it think the values are continous
X_train = train_set.iloc[:,0:-1]

y_test = test_set.iloc[:,-1].astype(int)
X_test = test_set.iloc[:,0:-1]

# SVM
SVM = svm.SVC(decision_function_shape="ovo")
SVM.fit(X_train, y_train)
SVM.predict(X_test)
SVM_score = round(SVM.score(X_test, y_test), 3)
print(SVM_score)

# KNN
KNN=KNeighborsClassifier()
KNN.fit(X_train,y_train)
KNN.predict(X_test)
KNN_score = round(KNN.score(X_test, y_test), 3)
print(KNN_score)

# NB
NB = GaussianNB()
NB.fit(X_train,y_train)
NB.predict(X_test)
NB_score = round(NB.score(X_test, y_test), 3)
print(NB_score)

""" Implement code here that runs these models for different numbers of columns and rows and see how it scales time-wise """