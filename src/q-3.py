# Run the code in the following format: python q-3.py --test <./path of the test file>

import csv
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import pdb
import sys
import matplotlib.pyplot as plt
import argparse

def load_data (data):
    data = pd.read_csv(data)
    # data = data.iloc[:20,:]
    trainData, valData = train_test_split(data, test_size = 0.2, random_state = 42)

    y_val = valData.iloc[:,-1]
    x_val = valData.iloc[:,:-1]
    x_val = x_val.drop('Serial No.', axis = 1)

    y_train = trainData.iloc[:,-1]
    x_train = trainData.iloc[:,:-1]
    x_train = x_train.drop('Serial No.', axis = 1)

    return x_val,y_val,x_train,y_train

def load_test_data(args):
    data = pd.read_csv(args)
    x_test = data.drop('Serial No.', axis = 1)
    return x_test


def coefficients(x_train,y_train):
    x = x_train.values
    # print(x)
    temp = np.ones((x_train.shape[0],1))
    x=np.concatenate((temp,x),axis=1)
    
    a = np.linalg.inv(np.matmul(x.T,x))
    b = np.matmul(a,x.T)
    beta = np.matmul(b,y_train.values)
    print(beta)
    return beta

def prediction(x_test,beta):
    x = x_test.values
    temp = np.ones((x_test.shape[0],1))
    x=np.concatenate((temp,x),axis=1)
    y_pred = np.matmul(x,beta)
    return y_pred

def mae(y_pred,y_val):
    maError = []
    y_val = y_val.values
    for i in range(len(y_val)):
        maError.append(np.absolute(y_val[i]-y_pred[i]))
    return np.mean(maError)

def mse(y_pred,y_val):
    msError = []
    y_val = y_val.values
    for i in range(len(y_val)):
        msError.append(np.square(y_val[i]-y_pred[i]))
    return np.mean(msError)

def mpe(y_pred,y_val):
    mpError = []
    y_val = y_val.values
    for i in range(len(y_val)):
        mpError.append((y_val[i]-y_pred[i])/y_val[i])
    return np.mean(mpError)

def residualPlot(y_pred,y_val,x_val):
    for i in range(x_val.shape[1]):
        plt.figure()
        plt.scatter(x_val.iloc[:,i],(y_pred-y_val.values))
        plt.title("Residual Plot : Residue vs Feature Values of '%s'" % x_val.columns[i])
        plt.xlabel("Feature Values of '%s'" % x_val.columns[i])
        plt.ylabel("Residue (Prection-Actual)")
    plt.show()


if __name__ == "__main__":

    # Run the code in the following format: python q-3.py --test <./path of the test file>

    parser = argparse.ArgumentParser()
    parser.add_argument('--test',required = True, help = 'Enter the path of the test dataset')
    args = parser.parse_args()
    
    data = '../Dataset/AdmissionDataset/data.csv'

    x_val,y_val,x_train,y_train = load_data(data)
    x_test = load_test_data(str(args.test))
    # trainData.head()
    

    beta = coefficients(x_train,y_train)

    print("--------------------Training Completed--------------------")
    print('')

    y_pred = prediction(x_val,beta)

    print('Validation MAE : %s' % mae(y_pred,y_val))
    print('Validation MSE : %s' % mse(y_pred,y_val))
    print('Validation MPE : %s' % mpe(y_pred,y_val))
    print("--------------------Validation Completed--------------------")
    print('')

    y_pred_test = prediction(x_test,beta)
    
    print("--------------------Testing Completed--------------------")
    print(y_pred_test)
    # Uncomment to see the predicted values


    residualPlot(y_pred,y_val,x_val)

