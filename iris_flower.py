import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

data = pd.read_csv('iris.data.csv', header = None)
data.rename(columns = {0: 'sepal length (cm)', 1: 'sepal width (cm)', 2: 'petal length (cm)', 3:  'petal width (cm)', 4: 'names'}, inplace = True)
print(data.head())

x = data.drop('names', axis = 1)
y = data.names

encoder = LabelEncoder()
y = encoder.fit_transform(y)

# Split data into train and test
x_train , x_test, y_train, y_test = train_test_split(x,y, test_size= 0.2, stratify = y)

# create dataframe for train data and test data.
train_data = pd.concat([x_train, pd.Series(y_train)], axis =1)
test_data = pd.concat([x_test, pd.Series(y_test)], axis =1)

# Model Creation
logreg = RandomForestClassifier()
logRegFitted = logreg.fit(x_train, y_train)
y_pred = logRegFitted.predict(x_test)

# acc = LogReg.score(y_pred, y)

# heatmap = sns.heatmap(data.corr(), cmap = 'BuPu', annot = True)
import streamlit as st
# Save the model
import joblib
joblib.dump(logreg, 'Logistic_Model.pkl')

import streamlit as st
from sklearn.datasets import load_iris

st.write("""
# Simple Iris Flower Prediction App
This app predicts the ***Iris Flower*** type!
""")

st.sidebar.header('User Input Parameters')



def user_input_features():
    sepal_length = st.sidebar.slider('Sepal length', 4.3, 7.9, 5.4)
    sepal_width = st.sidebar.slider('Sepal width', 2.0, 4.4, 3.4)
    petal_length = st.sidebar.slider('Petal length', 1.0, 6.9, 1.3)
    petal_width = st.sidebar.slider('Petal width', 0.1, 2.5, 0.2)
    data = {'sepal_length': sepal_length,
            'sepal_width': sepal_width,
            'petal_length': petal_length,
            'petal_width': petal_width}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

st.subheader('User Input parameters')
st.write(df)

iris = load_iris()
X = iris.data
Y = iris.target

clf = RandomForestClassifier()
clf.fit(X, Y)

prediction = clf.predict(df)
prediction_proba = clf.predict_proba(df)

st.subheader('Class labels and their corresponding index number')
st.write(iris.target_names)

st.subheader('Prediction')
st.write(iris.target_names[prediction])
# st.write(prediction)


st.subheader('Prediction Probability')
st.write(prediction_proba)
