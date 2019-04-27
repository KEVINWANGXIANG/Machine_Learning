from numpy import *
import numpy as np

#载入数据
def loadDataSet(fileName):
    numFeat = len(open(fileName).readline().split('\t')) - 1
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat, labelMat

#计算回归系数
def standRegres(xArr, yArr):
    xMat = mat(xArr)
    yMat = mat(yArr).T
    xTx = xMat.T * xMat
    #计算行列式是否是0
    if linalg.det(xTx) == 0.0:
        print("This matrix is singular, cannot do inverse")
        return
    ws = xTx.I * (xMat.T * yMat)
    return ws

xArr, yArr = loadDataSet(r'F:/Python/机器学习实战/MLiA_SourceCode/machinelearninginaction/Ch08/ex0.txt')
# print(xArr[0:2])
ws = standRegres(xArr, yArr)
# print(ws)
xMat = mat(xArr)
yMat = mat(yArr)
#预测的值
yHat = xMat * ws

#绘图
import matplotlib.pyplot as plt

# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.scatter(xMat[:, 1].flatten().A[0], yMat.T[:, 0].flatten().A[0])

# xCopy = xMat.copy()
# xCopy.sort(0)
# print(xCopy)
# yHat = xCopy * ws
# ax.plot(xCopy[:, 1], yHat)
# plt.show()
# yHat = xMat * ws
# print(corrcoef(yHat.T, yMat))

def lwlr(testPoint, xArr, yArr, k=1.0):
    xMat = mat(xArr)
    yMat = mat(yArr).T
    m = shape(xMat)[0]
    #返回m * m的单位矩阵
    weights = mat(eye((m)))
    for j in range(m):
        diffMat = testPoint - xMat[j, :]
        weights[j, j] = np.exp(diffMat * diffMat.T / (-2 * k ** 2))
    xTx = xMat.T * (weights * xMat)
    if linalg.det(xTx) == 0.0:
        print("This matrix is singular, cannot do inverse")
        return
    ws = xTx.I * (xMat.T * (weights * yMat))
    return testPoint * ws

def lwlrTest(testArr, xArr, yArr, k=1.0):
    m = shape(testArr)[0]
    yHat = np.zeros(m)
    for i in range(m):
        yHat[i] = lwlr(testArr[i], xArr, yArr, k)
    return yHat

xArr, yArr = loadDataSet(r'F:/Python/机器学习实战/MLiA_SourceCode/machinelearninginaction/Ch08/ex0.txt')
# print(yArr[0])
# print(lwlr(xArr[0], xArr, yArr, 1.0))
# print(lwlr(xArr[0], xArr, yArr, 0.001))
# print(lwlr(xArr[0], xArr, yArr, 0.003))

yHat = lwlrTest(xArr, xArr, yArr, 0.01)
# print(yHat)

xMat = mat(xArr)

srtInd = xMat[:, 1].argsort(0)
xSort = xMat[srtInd][:, 0, :]

import matplotlib.pyplot as plt
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.plot(xSort[:, 1], yHat[srtInd])
# ax.scatter(xMat[:, 1].flatten().A[0], mat(yArr).T.flatten().A[0], s=2, c='red')
# plt.show()

#计算误差
def rssError(yArr, yHatArr):
    return ((yArr - yHatArr) ** 2).sum()

abX, abY = loadDataSet(r'F:/Python/机器学习实战/MLiA_SourceCode/machinelearninginaction/Ch08/abalone.txt')
# yHat01 = lwlrTest(abX[0:99], abX[0:99], abY[0:99], 0.1)
# yHat1 = lwlrTest(abX[0:99], abX[0:99], abY[0:99], 1)
# yHat10 = lwlrTest(abX[0:99], abX[0:99], abY[0:99], 10)

# print(rssError(abY[0:99], yHat01.T))
# print(rssError(abY[0:99], yHat1.T))
# print(rssError(abY[0:99], yHat10.T))

# yHat01 = lwlrTest(abX[100:199], abX[0:99], abY[0:99], 0.1)
# yHat1 = lwlrTest(abX[100:199], abX[0:99], abY[0:99], 1)
# yHat10 = lwlrTest(abX[100:199], abX[0:99], abY[0:99], 10)
#
# print(rssError(abY[100:199], yHat01.T))
# print(rssError(abY[100:199], yHat1.T))
# print(rssError(abY[100:199], yHat10.T))


# ws = standRegres(abX[0:99], abY[0:99])
# yHat = mat(abX[100:199]) * ws
# error = rssError(abY[100:199], yHat.T.A)
# print(error)

#岭回归
def ridgeRegres(xMat, yMat, lam=0.2):
    xTx = xMat.T * xMat
    #eye生成单位矩阵
    denom = xTx + eye(shape(xMat)[1]) * lam
    if linalg.det(denom) == 0.0:
        print("this matrix is singular, cannot do inverse")
        return
    ws = denom.I * (xMat.T * yMat)
    return ws

#多个测试
def ridgeTest(xArr, yArr):
    xMat = mat(xArr)
    yMat = mat(yArr).T
    #0代表取各列的平均值
    #标准化
    yMean = mean(yMat, 0)
    yMat = yMat - yMean
    xMean = mean(xMat, 0)
    xVar = var(xMat, 0)
    xMat = (xMat - xMean) / xVar
    numTestPts = 30
    wMat = zeros((numTestPts, shape(xMat)[1]))
    for i in range(numTestPts):
        ws = ridgeRegres(xMat, yMat, np.exp(i-10))
        wMat[i, :] = ws.T
    return wMat

# abX, abY = loadDataSet(r'F:/Python/机器学习实战/MLiA_SourceCode/machinelearninginaction/Ch08/abalone.txt')
# ridgeWeights = ridgeTest(abX, abY)
# print(ridgeWeights)
# print(len(abX))
#绘图
import matplotlib.pyplot as plt

# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.plot(ridgeWeights)
# plt.show()
def regularize(xMat):#regularize by columns
    inMat = xMat.copy()
    inMeans = mean(inMat, 0)   #calc mean then subtract it off
    inVar = var(inMat, 0)      #calc variance of Xi then divide by it
    inMat = (inMat - inMeans)/inVar
    return inMat

#前向逐步线性法回归
def stageWise(xArr, yArr, eps=0.01, numIt=100):
    xMat = mat(xArr)
    yMat = mat(yArr).T
    yMean = mean(yMat, 0)
    yMat = yMat - yMean
    xMat = regularize(xMat)
    m, n = shape(xMat)
    returnMat = zeros((numIt, n))
    ws = zeros((n, 1))
    wsTest = ws.copy()
    wsMax = ws.copy()
    for i in range(numIt):
        print(ws.T)
        lowestError = inf
        for j in range(n):
            for sign in [-1, 1]:
                wsTest = ws.copy()
                wsTest[j] += eps * sign
                yTest = xMat * wsTest
                rssE = rssError(yMat.A, yTest.A)
                if rssE < lowestError:
                    lowestError = rssE
                    wsMax = wsTest
        ws = wsMax.copy()
        returnMat[i, :] = ws.T
    return returnMat

xArr, yArr = loadDataSet(r'F:/Python/机器学习实战/MLiA_SourceCode/machinelearninginaction/Ch08/abalone.txt')

# returnMat = stageWise(xArr, yArr, 0.005, 1000)
# print(returnMat)
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.plot(returnMat)
# plt.show()
# [ 0.043 -0.011  0.12  ... -0.963 -0.105  0.187]]
#[[ 0.0430442  -0.02274163  0.13214087  0.02075182  2.22403814 -0.99895312 -0.11725427  0.16622915]]

# #线性回归的结果比较
# xMat = mat(xArr)
# yMat = mat(yArr).T
# xMat = regularize(xMat)
# yM = mean(yMat, 0)
# yMat = yMat - yM
# weights = standRegres(xMat, yMat.T)
# print(weights.T)


from time import sleep
import json
from urllib.request import urlopen
def searchForSet(retX, retY, setNum, yr, numPce, origPrc):
    sleep(10)
    myAPIstr = 'AIzaSyD2cR2KFyx12hXu6PFU-wrWot3NXvko8vY'
    searchURL = 'https://www.googleapis.com/shopping/search/v1/public/products?key=%s&country=US&q=lego+%d&alt=json' % (myAPIstr, setNum)
    pg = urlopen(searchURL)
    retDict = json.loads(pg.read())
    for i in range(len(retDict['items'])):
        try:
            currItem = retDict['items'][i]
            if currItem['product']['condition'] == 'new':
                newFlag = 1
            else: newFlag = 0
            listOfInv = currItem['product']['inventories']
            for item in listOfInv:
                sellingPrice = item['price']
                if sellingPrice > origPrc * 0.5:
                    print("%d\t%d\t%d\t%f\t%f" % (yr, numPce, newFlag, origPrc, sellingPrice))
                    retX.append([yr, numPce, newFlag, origPrc])
                    retY.append(sellingPrice)
        except: print('problem with item %d' % i)

def setDataCollect(retX, retY):
    searchForSet(retX, retY, 8288, 2006, 800, 49.99)
    searchForSet(retX, retY, 10030, 2002, 3096, 269.99)
    searchForSet(retX, retY, 10179, 2007, 5195, 499.99)
    searchForSet(retX, retY, 10181, 2007, 3428, 199.99)
    searchForSet(retX, retY, 10189, 2008, 5922, 299.99)
    searchForSet(retX, retY, 10196, 2009, 3263, 249.99)

# lgX = []
# lgY = []
# setDataCollect(lgX, lgY)

#交叉验证测试岭回归
def crossValidation(xArr, yArr, numVal=10):
    m = len(yArr)
    indexList = range(m)
    #误差矩阵
    errorMat = zeros((numVal, 30))
    for i in range(numVal):
        trainX = []
        trainY = []
        testX = []
        testY = []
        random.shuffle(indexList)
        for j in range(m):
            if j < m * 0.9:
                trainX.append(xArr[indexList[j]])
                trainY.append(yArr[indexList[j]])
            else:
                testX.append(xArr[indexList[j]])
                testY.append(yArr[indexList[j]])
        wMat = ridgeTest(trainX, trainY)
        for k in range(30):
            matTestX = mat(testX)
            matTrainX = mat(trainX)
            meanTrain = mean(matTrainX, 0)
            varTrain = var(matTrainX, 0)
            matTestX = (matTestX - meanTrain) / varTrain
            yEst = matTestX * mat(wMat[k, :]).T + mean(trainY)
            errorMat[i, k] = rssError(yEst.T.A, array(testY))
    meanErrors = mean(errorMat, 0)
    minMean = float(min(meanErrors))
    bestWeights = wMat[nonzero(meanErrors == minMean)]
    xMat = mat(xArr)
    yMat = mat(yArr).T
    meanX = mean(xMat, 0)
    varX = var(xMat, 0)
    #数据还原
    unReg = bestWeights / varX
    #y = a* x + b unreg->a,底下的那个相当于b
    print("the best model from Ridge Regression is:\n", unReg)
    print("with constant term:", -1 * sum(multiply(meanX, unReg)) + mean(yMat))



















