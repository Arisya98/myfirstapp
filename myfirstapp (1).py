import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score


st.title("Heart Failure Prediction")

st. markdown("""
This app simply performs prediction on heart failure clinical datasets
* **Data Source:** [Heart-Failure-Prediction.kaggle](https://www.kaggle.com/andrewmvd/heart-failure-clinical-data)
* **Reference:** Machine learning can predict survival of patients with heart failure from serum creatinine and ejection fraction alone. [Here](https://doi.org/10.1186/s12911-020-1023-5)
""" )


df = pd.read_csv('heart_failure.csv')
df = df.astype({"age": int})
st.dataframe(df)


st.write('The dimension of data is', str(df.shape[0]), 'rows and', str(df.shape[1]), 'columns')
#st.write('missing value =', str(df.isnull().sum()))


st.sidebar. write('Prepared by: Fatin Arisya Binti Azhar')

#sidebar.selectbox

st.sidebar.header('User Input Parameters')


option = st.sidebar.selectbox(
    'Select a classifier',
     ['KNN','Logistic Regression','SVM','Gaussian Naive Bayes','Random Forest'])


#sidebar.numberinput

values = st.sidebar.number_input('Pick a number', 0, 200)

X = df.drop(['DEATH_EVENT'], axis = 1)
y = df['DEATH_EVENT']
from sklearn.model_selection import train_test_split
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, random_state = values)

#classifier

if option=='KNN':
    knn = KNeighborsClassifier()
    knn.fit(Xtrain, ytrain)
    ypred = knn.predict(Xtest)


elif option=='Logistic Regression':
    logreg = LogisticRegression()
    logreg.fit(Xtrain, ytrain)
    ypred = logreg.predict(Xtest)


elif option=='SVM':
    svc = SVC()
    svc.fit(Xtrain, ytrain)
    ypred = svc.predict(Xtest)


elif option=='Gaussian Naive Bayes':
    nb = GaussianNB()
    nb.fit(Xtrain, ytrain)
    ypred = nb.predict(Xtest)


else:
    RF = RandomForestClassifier()
    RF.fit(Xtrain, ytrain)
    ypred = RF.predict(Xtest)



st.write('Confusion Matrix:', str(confusion_matrix(ytest, ypred)))
st.write('Accuracy: ', str(accuracy_score(ytest, ypred)))
st.write('Precision: ', str(precision_score(ytest, ypred)))
st.write('Recall/Sensitivity: ', str(recall_score(ytest, ypred)))
st.write('f1 Score: ', str(f1_score(ytest, ypred)))
st.write('ROC AUC: ', str(roc_auc_score(ytest, ypred)))   
st.write('classification report:', str(classification_report(ytest, ypred)))
