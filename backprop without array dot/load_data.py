import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
import seaborn as sns
from scipy import stats
from scipy.stats import shapiro


"""Load data set and print info about it """
df = pd.read_csv('creditcard.csv')
print(df.info())
print(df['Class'].value_counts())
print(df.isnull().sum())


"""normalize data set"""
max_abs = df.abs().max()
df = df / max_abs
print(df)
print (df.max())
print (df.min())



"""Separate data into training and testing groups 70 , 30 """
Y = df['Class']
X = df.drop('Class', axis=1)
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.4
)
print("X_train : ", X_train.shape)
print("X_test : " , X_test.shape)
print("Y_train : " , Y_train.shape)
print("Y_test : " , Y_test.shape)
print("Y_train value : \n" , Y_train.value_counts())
print("Y_test value : \n" , Y_test.value_counts())

