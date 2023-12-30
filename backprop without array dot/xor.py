import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from dense import Dense
from activations import Tanh
from losses import mse, mse_prime
from network import train, predict
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
Y = np.array(df['Class'])
X = np.array(df.drop('Class', axis=1))
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.4
)

Y_train = Y_train.reshape(Y_train.shape[0],1)
Y_test = Y_test.reshape(Y_test.shape[0],1)

X_train = X_train.reshape(X_train.shape[0],X_train.shape[1],1)
Y_train = Y_train.reshape(Y_train.shape[0],Y_train.shape[1],1)

print("X_train : ", X_train.shape)
print("X_test : " , X_test.shape)
print("Y_train : " , Y_train.shape)
print("Y_test : " , Y_test.shape)


X=X_train
Y=Y_train

network = [
    Dense(30, 3),
    Tanh(),
    Dense(3, 1),
    Tanh()
]

# train
train(network, mse, mse_prime, X, Y, epochs=10, learning_rate=0.1)
