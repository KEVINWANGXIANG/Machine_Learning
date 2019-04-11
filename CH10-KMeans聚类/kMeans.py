from numpy import *
from math import *
import random
import numpy as np


'''
def loadDataSet(fileName):
    dataMat = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        #map()函数将curLine都变成了浮点型数据，高阶函数
        fltLine = list(map(float, curLine))
        dataMat.append(fltLine)
    return dataMat


#计算两个向量的欧式距离
def distEclud(vecA, vecB):
    return sqrt(sum(power(vecA - vecB, 2)))


#构建初始的K个质心
def randCent(dataSet, k):
    n = shape(dataSet)[1]
    centroids = mat(np.zeros((k, n)))
    for j in range(n):
        #列的最小值
        minJ = min(dataSet[:, j])
        rangeJ = float(max(dataSet[:, j]) - minJ)
        centroids[:, j] = mat(minJ + rangeJ * np.random.rand(k, 1))
    return centroids
# datMat = []
# datList = []
# for line in open(r'F:\Python\机器学习实战\MLiA_SourceCode\machinelearninginaction\Ch10\places.txt').readlines():
#     lineArr = line.split('\t')
#     datList.append([float(lineArr[4]), float(lineArr[3])])
#     datMat = mat(datList)
# # dataMat = np.mat(loadDataSet(r'F:/Python/机器学习实战/MLiA_SourceCode/machinelearninginaction/Ch10/testSet.txt'))
# print(datMat)
# centroids = randCent(datMat, 2)
# print(centroids)


def kMeans(dataSet, k, distMeas=distEclud, createCent=randCent):
    m = np.shape(dataSet)[0]
    clusterAssment = np.mat(np.zeros((m, 2)))
    centroids = createCent(dataSet, k)
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        for i in range(m):
            minDist = inf
            minIndex = -1
            for j in range(k):
                distJI = distMeas(centroids[j, :], dataSet[i, :])
                if distJI < minDist:
                    minDist = distJI
                    minIndex = j
            if clusterAssment[i, 0] != minIndex:
                clusterChanged = True
            clusterAssment[i, :] = minIndex, minDist ** 2
        print(centroids)
        for cent in range(k):
            #nonzero函数是numpy中用于得到数组array中非零元素的位置（数组索引）的函数。
            ptsInclust = dataSet[nonzero(clusterAssment[:, 0].A == cent)[0]]
            centroids[cent, :] = mean(ptsInclust, axis=0)
    return centroids, clusterAssment

# myCentroids, clustAssing = kMeans(dataMat, 4)
# print(myCentroids)
# print(clustAssing)

#二分K-均值聚类算法
def biKmeans(dataSet, k, distMeas=distEclud):
    m = np.shape(dataSet)[0]
    #初始化簇索引矩阵值
    clusterAssment = np.mat(np.zeros((m, 2)))
    #创建一个初始簇
    centroid0 = np.mean(dataSet, axis=0).tolist()[0]
    centList = [centroid0]
    #只有一个簇索引矩阵的值
    for j in range(m):
        clusterAssment[j, 1] = distMeas(np.mat(centroid0), dataSet[j, :]) ** 2
    while (len(centList) < k):
        lowestSSE = inf
        #对每个簇计算总误差
        for i in range(len(centList)):
            ptsInCurrCluster = dataSet[np.nonzero(clusterAssment[:, 0].A == i)[0], :]
            #给定的簇分为两类
            centroidMat, splitClustAss = kMeans(ptsInCurrCluster, 2, distMeas)
            sseSplit = sum(splitClustAss[:, 1])
            sseNotSplit = sum(clusterAssment[nonzero(clusterAssment[:, 0].A != i)[0], 1])
            print("sseSplit, and sseNotSplit", sseSplit, sseNotSplit)
            #选择总误差最小的那个簇
            if (sseSplit + sseNotSplit) < lowestSSE:
                bestCentToSplit = i
                bestNewCents = centroidMat
                bestClustAss = splitClustAss.copy()
                lowestSSE = sseSplit + sseNotSplit
        bestClustAss[nonzero(bestClustAss[:, 0].A == 1)[0], 0] = len(centList)
        bestClustAss[nonzero(bestClustAss[:, 0].A == 0)[0], 0] = bestCentToSplit
        print("the bestCentToSplit is:", bestCentToSplit)
        print("the len of bestClustAss is:", len(bestClustAss))
        centList[bestCentToSplit] = bestNewCents[0, :]
        centList.append(bestNewCents[1, :].tolist()[0])
        clusterAssment[nonzero(clusterAssment[:, 0].A == bestCentToSplit)[0], :] = bestClustAss
    return centList, clusterAssment

# dataMat3 = mat(loadDataSet(r'F:/Python/机器学习实战/MLiA_SourceCode/machinelearninginaction/Ch10/testSet2.txt'))
# centList, myNewAssments = biKmeans(dataMat3, 3)
# centList = np.mat(centList)
# print(centList)
# print(myNewAssments)
# print(len(myNewAssments))

from math import *
#计算地球表面两点之间的距离
def distSLC(vecA, vecB):
    a = sin(vecA[0, 1] * pi / 180) * sin(vecB[0, 1] * pi / 180)
    b = cos(vecA[0, 1] * pi / 180) * cos(vecB[0, 1] * pi / 180) * cos(pi * (vecB[0, 0] - vecA[0, 0]) / 180)
    return np.arccos(a + b) * 6371.0

import matplotlib
import matplotlib.pyplot as plt

def clusterClubs(numClust=5):
    datList = []
    for line in open(r'F:\Python\机器学习实战\MLiA_SourceCode\machinelearninginaction\Ch10\places.txt').readlines():
        lineArr = line.split('\t')
        datList.append([float(lineArr[4]), float(lineArr[3])])
        datMat = mat(datList)
        print(datMat)
        myCentroids, clustAssing = biKmeans(datMat, numClust, distMeas=distSLC)
        fig = plt.figure()
        rect = [0.1, 0.1, 0.8, 0.8]
        scatterMarkers = ['s', 'o', '^', '8', 'p', 'd', 'v', 'h', '>', '<']
        axprops = dict(xticks=[], yticks=[])
        ax0 = fig.add_axes(rect, label='ax0', **axprops)
        imgP = plt.imread(r'F:/Python/机器学习实战/MLiA_SourceCode/machinelearninginaction/Ch10/Portland.png')
        ax0.imshow(imgP)
        ax1 = fig.add_axes(rect, label="ax1", frameon=False)
        for i in range(numClust):
            ptsInCurrCluster = datMat[nonzero(clustAssing[:, 0].A == i)[0], :]
            markerStyle = scatterMarkers[i % len(scatterMarkers)]
            ax1.scatter(ptsInCurrCluster[:, 0].flatten().A[0], ptsInCurrCluster[:, 1].flatten().A[0], marker=markerStyle, s=90)
        ax1.scatter(myCentroids[:, 0].flatten().A[0], myCentroids[:, 1].flatten().A[0], marker="+", s=300)
        plt.show()
clusterClubs(5)
'''

from numpy import *
def loadDataSet(fileName):      #general function to parse tab -delimited floats
    dataMat = []                #assume last column is target value
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = map(float, curLine) #map all elements to float()
        dataMat.append(fltLine)
    return dataMat

def distEclud(vecA, vecB):
    return sqrt(sum(power(vecA - vecB, 2))) #la.norm(vecA-vecB)

def randCent(dataSet, k):
    n = shape(dataSet)[1]
    centroids = mat(zeros((k, n)))#create centroid mat
    for j in range(n):#create random cluster centers, within bounds of each dimension
        minJ = min(dataSet[:, j])
        rangeJ = float(max(dataSet[:, j]) - minJ)
        centroids[:, j] = mat(minJ + rangeJ * random.rand(k, 1))
    return centroids

def kMeans(dataSet, k, distMeas=distEclud, createCent=randCent):
    m = shape(dataSet)[0]
    clusterAssment = mat(zeros((m, 2)))#create mat to assign data points
                                      #to a centroid, also holds SE of each point
    centroids = createCent(dataSet, k)
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        for i in range(m):#for each data point assign it to the closest centroid
            minDist = inf
            minIndex = -1
            for j in range(k):
                distJI = distMeas(centroids[j, :], dataSet[i, :])
                if distJI < minDist:
                    minDist = distJI; minIndex = j
            if clusterAssment[i, 0] != minIndex:
                clusterChanged = True
            clusterAssment[i, :] = minIndex, minDist**2
        print(centroids)
        for cent in range(k):#recalculate centroids
            ptsInClust = dataSet[nonzero(clusterAssment[:, 0].A==cent)[0]]#get all the point in this cluster
            centroids[cent, :] = mean(ptsInClust, axis=0) #assign centroid to mean
    return centroids, clusterAssment

def biKmeans(dataSet, k, distMeas=distEclud):
    m = shape(dataSet)[0]
    clusterAssment = mat(zeros((m, 2)))
    centroid0 = mean(dataSet, axis=0).tolist()[0]
    centList =[centroid0] #create a list with one centroid
    for j in range(m):#calc initial Error
        clusterAssment[j, 1] = distMeas(mat(centroid0), dataSet[j, :])**2
    while (len(centList) < k):
        lowestSSE = inf
        for i in range(len(centList)):
            ptsInCurrCluster = dataSet[nonzero(clusterAssment[:, 0].A == i)[0], :]#get the data points currently in cluster i
            centroidMat, splitClustAss = kMeans(ptsInCurrCluster, 2, distMeas)
            sseSplit = sum(splitClustAss[:, 1])#compare the SSE to the currrent minimum
            sseNotSplit = sum(clusterAssment[nonzero(clusterAssment[:, 0] != i)[0], 1])
            print("sseSplit, and notSplit: ", sseSplit, sseNotSplit)
            if (sseSplit + sseNotSplit) < lowestSSE:
                bestCentToSplit = i
                bestNewCents = centroidMat
                bestClustAss = splitClustAss.copy()
                lowestSSE = sseSplit + sseNotSplit
        bestClustAss[nonzero(bestClustAss[:, 0].A == 1)[0], 0] = len(centList) #change 1 to 3,4, or whatever
        bestClustAss[nonzero(bestClustAss[:, 0].A == 0)[0], 0] = bestCentToSplit
        print('the bestCentToSplit is: ', bestCentToSplit)
        print('the len of bestClustAss is: ', len(bestClustAss))
        centList[bestCentToSplit] = bestNewCents[0, :].tolist()[0]#replace a centroid with two best centroids
        centList.append(bestNewCents[1, :].tolist()[0])
        clusterAssment[nonzero(clusterAssment[:, 0].A == bestCentToSplit)[0], :]= bestClustAss#reassign new clusters, and SSE
    return mat(centList), clusterAssment

def distSLC(vecA, vecB):#Spherical Law of Cosines
    a = sin(vecA[0, 1]*pi/180) * sin(vecB[0, 1]*pi/180)
    b = cos(vecA[0, 1]*pi/180) * cos(vecB[0, 1]*pi/180) * \
                      cos(pi * (vecB[0, 0]-vecA[0, 0]) /180)
    return arccos(a + b)*6371.0 #pi is imported with numpy

import matplotlib
import matplotlib.pyplot as plt
def clusterClubs(numClust=5):
    datList = []
    for line in open('F:\Python\机器学习实战\MLiA_SourceCode\machinelearninginaction\Ch10\places.txt').readlines():
        lineArr = line.split('\t')
        datList.append([float(lineArr[4]), float(lineArr[3])])
    datMat = mat(datList)
    myCentroids, clustAssing = biKmeans(datMat, numClust, distMeas=distSLC)
    fig = plt.figure()
    rect=[0.1, 0.1, 0.8, 0.8]
    scatterMarkers=['s', 'o', '^', '8', 'p', \
                    'd', 'v', 'h', '>', '<']
    axprops = dict(xticks=[], yticks=[])
    ax0=fig.add_axes(rect, label='ax0', **axprops)
    imgP = plt.imread('F:\Python\机器学习实战\MLiA_SourceCode\machinelearninginaction\Ch10\Portland.png')
    ax0.imshow(imgP)
    ax1=fig.add_axes(rect, label='ax1', frameon=False)
    for i in range(numClust):
        ptsInCurrCluster = datMat[nonzero(clustAssing[:, 0].A == i)[0], :]
        markerStyle = scatterMarkers[i % len(scatterMarkers)]
        ax1.scatter(ptsInCurrCluster[:, 0].flatten().A[0], ptsInCurrCluster[:, 1].flatten().A[0], marker=markerStyle, s=90)
    ax1.scatter(myCentroids[:, 0].flatten().A[0], myCentroids[:, 1].flatten().A[0], marker='+', s=300)
    plt.show()
clusterClubs(5)















