# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 16:33:22 2023

@author: mmoein2
"""

# %%Importing Libraries needed to run the tool
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split #split data in training and testing set
from sklearn.model_selection import cross_val_score #K-fold cross validation
from sklearn import linear_model #Importing both linear regression and logistic regression
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import time
from datetime import timedelta
start_time = time.monotonic()
sns.set(style='darkgrid')

# %%Read the file
data = pd.read_csv("... .csv", header=0)

# %%Print number of rows and colums read
print("{0} rows and {1} columns".format(len(data.index), len(data.columns)))
print("")


# %% Removing rows without meaningful data points
data = data[data.wfh_pre != "Question not displayed to respondent"]
data = data[data.wfh_now != "Question not displayed to respondent"]
data = data[data.wfh_expect != "Question not displayed to respondent"]
data = data[data.jobcat_now_w1b != "Question not displayed to respondent"]
data = data[data.jobcat_now_w1b != "Variable not available in datasource"]

# %% MApping data into dummies
data['wfh_pre']= data['wfh_pre'].map({'Yes':1 ,'No' :0,'':0})
data['gender']= data['gender'].map({'Female':1 ,'Male':0,'':0})
data['wfh_now']= data['wfh_now'].map({'Yes':1 ,'No':0,'':0})

# %% Introducing data variables
X = np.column_stack((data.wfh_pre, data.wfh_now, data.age, data.gender,
                     pd.get_dummies(data.educ), pd.get_dummies(data.hhincome),
                     pd.get_dummies((data.jobcat_now_w1b))))
Y = data.wfh_expect

# %%
# %%
# %%
# %% Modeling
# %%
# %%
# %%


# %% Simple logistic regression

Y_LR = Y #define Y variable
X_scaled = preprocessing.scale(X)


LR = linear_model.LogisticRegression(multi_class='multinomial') #Call logistic regression model
LR.fit(X_scaled, Y_LR) #Fit regression
Y_pred = LR.predict(X_scaled) #Calculate points predicted to measure accuracy
Accuracy = metrics.accuracy_score(Y_LR, Y_pred) #Calculate accuracy score
print("Simple LR Accuracy: {0}".format(Accuracy))
print("")
print("SC1 LR Coefficients: {0}".format(LR.coef_))

# %% saving output
df = pd.DataFrame (LR.coef_)
df.to_csv('SC1LR.csv')

# %% Plotting the Confusion Matrix
confusion_matrix = metrics.confusion_matrix(Y_LR,Y_pred,normalize='true')#, display_labels=lr.classes_, cmap="Blues", normalize='true')
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])

cm_display.plot()
plt.savefig('Confusion_LRsimple.JPEG') #Saving the plot
plt.show()

# %%
# %%

# %% Logistic regression with train_test_split using Built in train test split function in sklearn

# %% Splitting the dataset to 80% trainig and 20% testing
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, stratify=Y)

Y_LR = Y_train #define Y variable

LR = linear_model.LogisticRegression(multi_class='multinomial') #Call logistic regression model
LR.fit(X_train, Y_LR) #Fit regression
Y_pred_train = LR.predict(X_train) #Calculate points predicted to measure accuracy
Y_pred_test = LR.predict(X_test) #Calculate points predicted to measure accuracy
Accuracy_train = metrics.accuracy_score(Y_LR, Y_pred_train) #Calculate accuracy score
Accuracy_test = metrics.accuracy_score(Y_test, Y_pred_test) #Calculate accuracy score
print("")
print("Training accuracy: {0} and testing accuracy: {1}".format(Accuracy_train.round(4), Accuracy_test.round(4)))
print("")
print("SC2 LR Coefficients: {0}".format(LR.coef_))
# %% saving output
df = pd.DataFrame (LR.coef_)
df.to_csv('SC2LR.csv')

# %% Plotting the Confusion Matrix for LR with train_test_split

confusion_matrix = metrics.confusion_matrix( Y_LR,Y_pred_train,normalize='true')#, display_labels=lr.classes_, cmap="Blues", normalize='true')
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])

cm_display.plot()
plt.savefig( 'Confusion_LR_sp.JPEG') #Saving the plot
plt.show()



# %% Logistic regression with train_test_split and cross validation
Y_LR = Y_train #define Y variable

# %% Defining the model
LR = linear_model.LogisticRegression(multi_class='multinomial') #Call logistic regression model

# %% Cross Validation (CV) process

for k in range (10,11,5):
    scores = cross_val_score(LR, X_train, Y_LR, cv=k) #actual cross-validation process
    print("")
    print(scores)
    print("Accuracy: {0} (+/- {1})".format(scores.mean().round(2), (scores.std() * 2).round(2)))
    print("")
    
    LR.fit(X_train, Y_LR) #Fit regression
    Y_pred_train = LR.predict(X_train) #Calculate points predicted to measure accuracy
    Y_pred_test = LR.predict(X_test) #Calculate points predicted to measure accuracy
    Accuracy_train = metrics.accuracy_score(Y_LR, Y_pred_train) #Calculate accuracy score
    Accuracy_test = metrics.accuracy_score(Y_test, Y_pred_test) #Calculate accuracy score
    print("Training accuracy with cross validation: {0} and testing accuracy with cross validation: {1}".format(Accuracy_train.round(4), Accuracy_test.round(4)))
    # %% Plotting the Confusion Matrix
    confusion_matrix = metrics.confusion_matrix( Y_LR,Y_pred_train,normalize='true')
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])
    
    cm_display.plot()
    plt.savefig('Confusion_LRall.JPEG') #Saving the plot
    plt.show()
    print("SC3 LR Coefficients: {0}".format(LR.coef_))
    # %% saving output
    df = pd.DataFrame (LR.coef_)
    df.to_csv('SC3LR.csv')
time_duration=[]
end_time= time.monotonic()
time_duration.append(end_time - start_time)
print(time_duration)

