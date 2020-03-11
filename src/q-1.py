# Run the code in the following format: python q-1.py --test <./path of the test file>

import csv
import numpy as np
import pandas as pd
import math
import sys
from sklearn.model_selection import train_test_split
import operator
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_recall_fscore_support
import matplotlib.pyplot as plt
import argparse

def load_data (data,args,test_args):

    
    if args == 1:
        data = pd.read_csv(data,delimiter = r'\s+',header=None)
        data = data.iloc[:, :-1]
        data = data[data.columns[::-1]]
        data.columns = range(data.shape[1])

        testData = pd.read_csv(test_args,delimiter = r'\s+',header=None)
        testData = testData.iloc[:, :-1]
        testData = testData[testData.columns[::-1]]
        testData.columns = range(testData.shape[1])

    elif args == 2:
        data = pd.read_csv(data,delimiter = r'\s+',header=None)
        data = data.iloc[:, :-1]
        data = data[data.columns[::-1]]
        data.columns = range(data.shape[1])

        testData = pd.read_csv(test_args,delimiter = r'\s+',header=None)
        testData = testData.iloc[:, :-1]
        testData = testData[testData.columns[::-1]]
        testData.columns = range(testData.shape[1])

    else:
        data = pd.read_csv(data,header=None)
        testData = pd.read_csv(test_args,header=None)

    trainData, valData = train_test_split(data, test_size = 0.2, random_state = 42)
 
    return trainData,valData,testData



def normalization(Data):
    for i in range(Data.shape[1]):
        Data[[i]] = (Data[[i]] - Data[[i]].mean())/Data[[i]].std()
 
    return Data


def euclideanDist(point_1,point_2,length):
    dist = 0
    for i in range (length):
        dist = dist + np.square(point_1[i] - point_2[i])
    # dist = float(dist)
    return np.sqrt(dist)


def manhattenDist(point_1,point_2,length):
    dist = 0
    for i in range(length):
        dist+= np.absolute(point_1[i] - point_2[i])
    return dist

def minkowskiDist(point_1,point_2,length):
    dist = 0
    p = 3
    for i in range (length):
        dist += np.power(np.absolute(point_1[i] - point_2[i]),p)
    # dist = float(dist)
    return np.power(dist,1/p)


def chebyshevDist(point_1,point_2,length):
    dist = []
    for i in range(length):
        dist.append(np.absolute(point_1[i] - point_2[i]))
    return np.amax(dist)


def cosineDist(point_1,point_2,length):
    dist = 0
    a,b,c= 0,0,0
    for i in range (length):
        a += point_1[i]*point_2[i]
        b += point_1[i]**2
        c += point_2[i]**2
    dist = 1 - (a/(np.sqrt(b) * np.sqrt(c)))
    # dist = float(dist)
    return dist

def kNN(trainData, x_val_instance, k,distMetrics):

    distances = {}
    # print('test instane',x_val_instance)
    length = x_val_instance.shape[0]
    # print('len',length)
    for x in range(len(trainData)):

        if (distMetrics == 'Euclidean'):
            dist = euclideanDist(x_val_instance,trainData.iloc[x],length)
        elif (distMetrics == 'Manhatten'):
            dist = manhattenDist(x_val_instance,trainData.iloc[x],length)
        elif (distMetrics == 'Minkowski'):
            dist = minkowskiDist(x_val_instance,trainData.iloc[x],length)
        elif (distMetrics == 'Chebyshev'):
            dist = chebyshevDist(x_val_instance,trainData.iloc[x],length)
        elif (distMetrics == 'Cosine'):
            dist = cosineDist(x_val_instance,trainData.iloc[x],length)
        else:
            raise Exception('Invalid Distance Metrics ! Available metrics are Euclidean, Manhatten, Minkowski, Chebyshev, Cosine')

        distances[x] = dist

    dist_sorted = sorted(distances.items(),key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(dist_sorted[x][0])
    classFreq = {}
    for x in range(len(neighbors)):
        # print('neigh of x',trainData.iloc[neighbors[x]].iloc[-1])
        response = trainData.iloc[neighbors[x]].iloc[-1]
        if response in classFreq:
            classFreq[response]+=1
        else:
            classFreq[response] = 1
    sortedFreq = sorted(classFreq.items(),key=operator.itemgetter(1),reverse=True)
    return(sortedFreq[0][0], neighbors)


def performance(y_pred,y_val):
    class_name = list(np.unique(y_val))
    y_val = list(y_val)
    cm = np.zeros((len(class_name),len(class_name)))
    # print(cm)

    for i in range(len(y_val)):
        a = class_name.index(y_val[i])
        p = class_name.index(y_pred[i])
        cm[a][p] += 1 

    recall = []
    precision = []
    f1_score = []
    col_sum = cm.sum(axis = 0)
    row_sum = cm.sum(axis = 1)
    for i in range(len(class_name)):
        recall.append(cm[i][i]/col_sum[i])
        precision.append(cm[i][i]/row_sum[i])
        f1_score.append((2*recall[i]*precision[i])/(recall[i]+precision[i]))
    recall = np.mean(recall)
    precision = np.mean(precision)
    f1_score = np.mean(f1_score)

    return recall,precision,f1_score


def kNN_sk(x_train,y_train,x_val,y_val,k):
    neigh_sk = KNeighborsClassifier(k)
    neigh_sk.fit(x_train,y_train)
    y_pred_sk = neigh_sk.predict(x_val)
    accuracy_sk = np.sum(y_val == y_pred_sk) / y_val.shape[0]
    precision_sk,recall_sk,f1_score_sk,x = precision_recall_fscore_support(y_val,y_pred_sk,average = 'weighted')
    print('Accuracy_sk: ', accuracy_sk)
    print('Recall_sk: ', recall_sk)
    print('Precision_sk: ', precision_sk)
    print('F1 Score_sk: ', f1_score_sk)

def plot_KvsAcc(k,trainData,x_val,y_val):
    # 
    for j in ('Euclidean','Manhatten','Minkowski','Chebyshev','Cosine'):
        accu = []

        for kk in range(1,k+1):
            y_pred = []
            for i in range(len(x_val)):
                result,neigh = kNN(trainData, x_val.iloc[i], kk, j)
                y_pred.append(result)
            accu.append(np.sum(y_val == y_pred) / y_val.shape[0])
        plt.plot(range(1,26),accu,label = j)
        plt.title("Accuracy vs 'k' in kNN Classifier for Various Distance Metrics")
        plt.xlabel("No. of Neighbors (k)")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.grid(True,linestyle='--')
    plt.show()


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--test',required = True, help = 'Enter the path of the test dataset')
    test_args = parser.parse_args()

    print('Enter the dataset: 1: Robot1, 2: Robot2, 3: Iris')
    args = int(input())

    if args == 1:
        data = '../Dataset/RobotDataset/Robot1'
    elif args == 2:
        data = '../Dataset/RobotDataset/Robot2'
    elif args == 3:
        data = '../Dataset/Iris/Iris.csv'
    else:
        raise Exception('Invalid Input for type of dataset')
        
    # data = './Iris/Iris.csv'

    trainData,valData,testData = load_data(data,args,str(test_args.test))
    # print('data',trainData)
    # trainData,valData = normalization(trainData,valData)
    # trainData.head()
    

    x_val = valData.iloc[:, :-1]
    # x_val = normalization(x_val)
    y_val = valData.iloc[:,-1]

    x_train = trainData.iloc[:, :-1]
    # x_train = normalization(x_train)
    y_train = trainData.iloc[:,-1]

    # testData = normalization(testData)


    k = 15
    y_pred = []

    for i in range(len(x_val)):
        result,neigh = kNN(trainData, x_val.iloc[i], k, 'Euclidean')
        # print('result',result)
        # print('neigh', neigh)
        y_pred.append(result)

    recall,precision,f1_score = performance(y_pred,y_val)
    accuracy = np.sum(y_val == y_pred) / y_val.shape[0]

    print('-----------Validation Complete-----------')
    print('')

    print('')
    print('-----------------Self made kNN model-----------------')
    print('Accuracy: ', accuracy)
    print('Recall: ', recall)
    print('Precision: ', precision)
    print('F1 Score: ', f1_score)
    print('')

    print('-----------------Scikit-Learn kNN model for verification-----------------')
    kNN_sk(x_train,y_train,x_val,y_val,k)

    k_range = 25
    # plot_KvsAcc(k_range,trainData,x_val,y_val)

    y_test = []
    # print(testData)
    # print(len(testData))

    for i in range(len(testData)):
        result,neigh = kNN(trainData, testData.iloc[i], k, 'Euclidean')
        y_test.append(result)

    print('-----------Testing Complete-----------')
    print('')
    print('Predicted Values:', y_test)