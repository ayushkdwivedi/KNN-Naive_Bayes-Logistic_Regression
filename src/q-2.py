# Run the code in the following format: python q-3.py --test <./path of the test file>

import csv
import numpy as np
import pandas as pd
import math
import sys
from sklearn.model_selection import train_test_split
import operator
import collections
import pdb
import argparse

def load_data (data):
    data = pd.read_csv(data)
    # data = data.iloc[:20,:]
    data.columns = ['ID','Age','Experience','AnnIncome','ZIPCode','FamilySize','Spending','Education','Mortgage','Output','SecAcc','CD','IB','CreditCard']
    data = data.reindex(['ID','Age','Experience','AnnIncome','ZIPCode','Spending','Mortgage','FamilySize','Education','SecAcc','CD','IB','CreditCard','Output'], axis=1)
    # print(data.iloc[:10,:])
    trainData, valData = train_test_split(data, test_size = 0.2, random_state = 42)

    y_val = valData.iloc[:,-1]
    x_val = valData.iloc[:, 1:]
    x_val = x_val.drop('Output', axis = 1)

    y_train = trainData.iloc[:,-1]
    x_train = trainData.iloc[:, 1:]
    x_train = x_train.drop('Output', axis = 1)
    return x_train,y_train,x_val,y_val

def load_test_data(args):
    data = pd.read_csv(args)
    data.columns = ['ID','Age','Experience','AnnIncome','ZIPCode','FamilySize','Spending','Education','Mortgage','SecAcc','CD','IB','CreditCard']
    x_test = data.iloc[:, 1:]
    return x_test

def calcProb(outcomes):
    length = len(outcomes)
    prob = dict(collections.Counter(outcomes))
    for i in prob.keys():
        prob[i] = np.log(prob[i] / float(length))
    return prob

def calcStats(points):
    mean_var = {}
    mean_var[0] = np.mean(points)
    mean_var[1] = np.std(points)
    # print(mean_var)
    return mean_var

def gaussianProb(statDict,x):
    var = float(statDict[1])**2
    denom = (2*np.pi*var)**.5
    num = np.exp(-(float(x)-float(statDict[0]))**2/(2*var))
    if num == 0:
        return 0
    else:
        return np.log(num/denom)


def naive_bayes(x_train, y_train):
    classes = np.unique(y_train)
    likelihoods = {}
    for clas in classes:
        likelihoods[clas] = collections.defaultdict(list)

    classProb = calcProb(y_train)
  
    for clas in classes:
        favourables = y_train.index[y_train == clas].tolist()
        favData = x_train.loc[favourables,:]
        for i in range(0,favData.shape[1]):
            likelihoods[clas][i] += list(favData.iloc[:,i])
    # print(likelihoods)
  
    for clas in classes:
        for i in range(0,x_train.shape[1]):
            if i<6:
                likelihoods[clas][i] = calcStats(likelihoods[clas][i])
            else:
                likelihoods[clas][i] = calcProb(likelihoods[clas][i])
    # print(likelihoods)
    return likelihoods,classes,classProb

def prediction (likelihoods,x_val,classes,classProb):
    results = {}
    for clas in classes:
        finalProb = classProb[clas]
        for i in range(0,len(x_val)):

            if i<6:
                indvLikelihood = gaussianProb(likelihoods[clas][i],x_val[i])
                finalProb += indvLikelihood
            else:
                indvLikelihood = likelihoods[clas][i]
                if x_val[i] in indvLikelihood.keys():
                    finalProb += indvLikelihood[x_val[i]]
                else:
                    finalProb += 0
        results[clas] = finalProb
    if results[0]>results[1]:
        return 0
    else:
        return 1
  
if __name__ == "__main__":

    # Run the code in the following format: python q-3.py --test <./path of the test file>
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--test',required = True, help = 'Enter the path of the test dataset')
    args = parser.parse_args()

    data = '../Dataset/LoanDataset/data.csv'

    x_train,y_train,x_val,y_val = load_data(data)
    x_test = load_test_data(str(args.test))

    likelihoods,classes,classProb = naive_bayes(x_train, y_train)
    print('-----------Training Complete-----------')
    print('')
    y_pred = []
    for i in range(len(x_val)):
        y_pred.append(prediction(likelihoods,x_val.iloc[i],classes,classProb))
    
    accuracy = np.sum(y_val == y_pred) / y_val.shape[0]
    print('Validation Accuracy: ',accuracy)
    print('-----------Validation Complete-----------')
    print('')

    y_pred_test = []
    for i in range(len(x_test)):
        y_pred_test.append(prediction(likelihoods,x_test.iloc[i],classes,classProb))
   
    
    print('-----------Testing Complete-----------')
    print(y_pred_test)
   # Uncomment to see the predicted values


    # xx = x_train['ZIPCode'] //There are 457 unique values of ZIPCode. Hence considering it as a categorical data//
    # for i in range(x_train.shape[1]):
    #     x = np.unique(x_train.iloc[:,i])
    #     print(len(x))