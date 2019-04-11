from numpy import *
import pandas
import numpy as np
def createDataSet(path):    # 创造示例数据
    fname = open(path)
    dataf = pandas.read_csv(fname)
    dataSet = dataf.iloc[:, 0:len(dataf.columns)].as_matrix()
    dataSet = list(dataSet)
    b = []
    for i in range(len(dataSet)):
        b.append(list(dataSet[i]))
    return b


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

fileName = r'C:/Users/Administrator/Desktop/kmeans_data.csv'
dataSet = createDataSet(fileName)
dataSet = np.mat(dataSet)
# print(dataSet)
# centList, clusterAssment = biKmeans(dataSet, 3)
# print(clusterAssment)


from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import matplotlib.font_manager as fm
import matplotlib.pylab as plt
# kms = KMeans(n_clusters=3, n_jobs=2, max_iter=500)
# y = kms.fit_predict(dataSet)
# print(y)
# K = range(1, 10)
# font = fm.FontProperties(fname=r"c:/windows/fonts/msyh.ttc", size=10)
# meandistortions = []
# for k in K:
#     kmeans = KMeans(n_clusters=k)
#     kmeans.fit(dataSet)
#     meandistortions.append(sum(np.min(cdist(dataSet, kmeans.cluster_centers_, 'euclidean'), axis=1)) / dataSet.shape[0])
# plt.plot(K, meandistortions, 'bx-')
# plt.xlabel('k')
# plt.ylabel('平均畸变程度', fontproperties=font)
# plt.title('用肘部法则来确定最佳的K值', fontproperties=font)
# plt.show()

print(dataSet)
import pandas as pd
# dataSet = pd.DataFrame(dataSet)
dataSet = pd.read_csv(r'C:/Users/Administrator/Desktop/kmeans_data.csv')
from pandas.plotting import radviz
plt.figure()
radviz(dataSet, 'label')








