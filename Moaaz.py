import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras.layers import Dense, Input, Activation, Dropout
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.models import Model, Sequential
from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import mnist
import seaborn as sns
from scipy import stats
from scipy.stats import shapiro


"""Load data set and print info about it """
df = pd.read_csv("creditcard.csv")
print(df.info())
print(df["Class"].value_counts())
print(df.isnull().sum())


"""normalize data set"""
max_abs = df.abs().max()
df = df / max_abs
print(df)
print(df.max())
print(df.min())


""""test if it distributed normally"""

# Describe the data
print(df.describe())

# Plot histograms
# for column in df.columns:
#    plt.figure(figsize=(10,4))
#    sns.histplot(df[column])
#    plt.title(f"Histogram of {column}")
#    plt.show()

# # Perform Q-Q Plots
# for column in df.columns:
#    plt.figure(figsize=(10,4))
#    stats.probplot(df[column], plot=plt)
#    plt.title(f"Q-Q Plot of {column}")
#    plt.show()


"""Separate data into training and testing groups 70 , 30 """
Y = df["Class"]
X = df.drop("Class", axis=1)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4)
print("X_train : ", X_train.shape)
print("X_test : ", X_test.shape)
print("Y_train : ", Y_train.shape)
print("Y_test : ", Y_test.shape)
print("Y_train value : \n", Y_train.value_counts())
print("Y_test value : \n", Y_test.value_counts())
