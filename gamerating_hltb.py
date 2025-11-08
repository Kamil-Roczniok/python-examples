import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,classification_report

#Load the Iris dataset
iris = load_iris()

# Features and target
X = iris.data
y = iris.target

#Split the data into 80% training 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

#Initialize the scaler
scaler = StandardScaler()

#Fit the scaler on the training data and transform it
X_train_scaled = scaler.fit_transform(X_train)

#Transforming the test data using the same scaler
X_test_scaled = scaler.transform(X_test)

#Initialise the Support Vector Classifier
model = SVC(kernel='linear')

#Train the model using the training data
model.fit(X_train_scaled, y_train)

#Make predidctions on the test set
y_pred = model.predict(X_test_scaled)

#Calculate the accuracy of the model
accuracy = accuracy_score(y_test,y_pred)

#Get a detailed classification report
print("\nClassification Report:")
print(classification_report(y_test,y_pred))



