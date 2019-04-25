import numpy as np
import pandas as pd
import math
from numpy import *

def loadSimpData():
    datMat = np.matrix([[1., 2.1], [2., 1.1], [1.3, 1.], [1., 1.], [2., 1.]])
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return datMat, classLabels

datMat, classLabels = loadSimpData()

#单层决策树的阈值过滤函数,将一列数据分为-1和1
def stumpClassify(dataMatrix, dimen, threshVal, threshIneq):
    #初始化元素都为1的矩阵
    retArray = np.ones((np.shape(dataMatrix)[0], 1))
    if threshIneq == 'lt': #将小于某一阈值的特征归类为-1，左为-1，右为1
        retArray[dataMatrix[:, dimen] <= threshVal] = -1.0
    else: #将大于某一阈值的特征归类为-1，左为-1，右为1
        retArray[dataMatrix[:, dimen] > threshVal] = -1.0
    return retArray

#找出具有最低错误率的单层决策树，D为权重矩阵
def buildStump(dataArr, classLabels, D):
    #变为矩阵
    dataMatrix = np.mat(dataArr)
    labelMat = np.mat(classLabels).T
    m, n = np.shape(dataMatrix)
    numSteps = 10.0
    #存储最佳决策树的相关信息
    bestStump = {}
    bestClasEst = np.mat(np.zeros((m, 1)))
    minError = np.inf
    for i in range(n):
        #某一列的最小值和最大值
        rangeMin = dataMatrix[:, i].min()
        rangeMax = dataMatrix[:, i].max()
        stepSize = (rangeMax - rangeMin) / numSteps
        #遍历各个步长区间
        for j in range(-1, int(numSteps) + 1):
            for inequal in ['lt', 'gt']:#两种阈值过滤模式
                threshVal = (rangeMin + float(j) * stepSize) #阈值计算公式
                #选定阈值后，调用阈值过滤函数分类预测
                predictedVals = stumpClassify(dataMatrix, i, threshVal, inequal)
                errArr = np.mat(np.ones((m, 1)))
                errArr[predictedVals == labelMat] = 0
                weightedError = D.T * errArr
                # print("split:dim %d, thresh %.2f, thresh ineqal: %s, the weighted error is %.3f" %(i, threshVal, inequal, weightedError))
                if weightedError < minError:
                    minError = weightedError
                    bestClasEst = predictedVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    #分别返回决策树相关信息，最小误差率，决策树预测输出结果
    return bestStump, minError, bestClasEst

D = np.mat(np.ones((5, 1)) / 5)
bestStump, minError, bestClasEst = buildStump(datMat, classLabels, D)
# print(bestStump)
# print(minError)
# print(bestClasEst)

#基于单层决策树的AdaBoost训练过程
def adaBoostTrainDS(dataArr, classLabels, numIt=40):
    #弱分类器
    weakClassArr = []
    m = np.shape(dataArr)[0]
    #初始权重
    D = np.mat(np.ones((m, 1)) / m)
    #记录每个数据点的类别估计累计值
    aggClassEst = np.mat(np.zeros((m, 1)))
    for i in range(numIt):
        bestStump, error, classEst = buildStump(dataArr, classLabels, D)
        # print("D", D.T)
        #计算弱分类器的系数
        alpha = float(0.5 * math.log((1.0 - error) / max(error, 1e-16)))
        # print(alpha)
        bestStump['alpha'] = alpha
        weakClassArr.append(bestStump)
        # print('classEst:', classEst.T)
        #更新权值分布
        #标签相同为-α，不同的类别标签为α
        expon = np.multiply(-1 * alpha * np.mat(classLabels).T, classEst)
        # print(expon)
        D = np.multiply(D, np.exp(expon))
        D = D / D.sum()
        aggClassEst += alpha * classEst
        # print("aggClassEst:", aggClassEst.T)
        aggErrors = np.multiply(np.sign(aggClassEst) != np.mat(classLabels).T, np.ones((m, 1)))
        errorRate = aggErrors.sum() / m
        print("total error:", errorRate, "\n")
        if errorRate == 0:
            break
    return weakClassArr

# classifierArray = adaBoostTrainDS(datMat, classLabels, 9)
# print(classifierArray)

#AdaBoost分类函数
def adaClassify(datToClass, classifierArr):
    dataMatrix = np.mat(datToClass)
    m = np.shape(dataMatrix)[0]
    aggClassEst = np.mat(np.zeros((m, 1)))
    for i in range(len(classifierArr)):
        classEst = stumpClassify(dataMatrix, classifierArr[i]['dim'], classifierArr[i]['thresh'], classifierArr[i]['ineq'])
        aggClassEst += classifierArr[i]['alpha'] * classEst
        # print(aggClassEst)
    return np.sign(aggClassEst)

# datArr, labelArr = loadSimpData()
# classifierArr = adaBoostTrainDS(datArr, labelArr, 30)
# # print(classifierArr)
# # labelPredict = adaClassify([0, 0], classifierArr)
# # print(labelPredict)
# labelPredict = adaClassify([[0, 0], [5, 5]], classifierArr)
# print(labelPredict)

#自适应数据加载函数
def loadDataSet(fileName):
    numFeat = len(open(fileName).readline().split('\t'))
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat - 1):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat, labelMat

datArr, labelArr = loadDataSet(r'F:/Python/机器学习实战/MLiA_SourceCode/machinelearninginaction/Ch07/horseColicTraining2.txt')
# print(datArr)
# print(len(datArr))
classifierArray = adaBoostTrainDS(datArr, labelArr, 10)
# print(classifierArray)
testArr, testLabelArr = loadDataSet(r'F:/Python/机器学习实战/MLiA_SourceCode/machinelearninginaction/Ch07/horseColicTest2.txt')
# print(len(testLabelArr))
prediction10 = adaClassify(testArr, classifierArray)
errArr = np.mat(np.ones((67, 1)))
print(errArr[prediction10 != np.mat(testLabelArr).T].sum())
print("Accuracy:%.5f" %(float(67 - errArr[prediction10 != np.mat(testLabelArr).T].sum()) / 67))


def plotROC(predStrengths, classLabels):
    import matplotlib.pyplot as plt
    cur = (1.0, 1.0) #cursor
    ySum = 0.0 #variable to calculate AUC
    numPosClas = sum(array(classLabels)==1.0)
    yStep = 1/float(numPosClas); xStep = 1/float(len(classLabels)-numPosClas)
    sortedIndicies = predStrengths.argsort()#get sorted index, it's reverse
    fig = plt.figure()
    fig.clf()
    ax = plt.subplot(111)
    #loop through all the values, drawing a line segment at each point
    for index in sortedIndicies.tolist()[0]:
        if classLabels[index] == 1.0:
            delX = 0
            delY = yStep
        else:
            delX = xStep
            delY = 0
            ySum += cur[1]
        #draw line from cur to (cur[0]-delX,cur[1]-delY)
        ax.plot([cur[0], cur[0]-delX], [cur[1], cur[1]-delY], c='b')
        cur = (cur[0]-delX, cur[1]-delY)
    ax.plot([0, 1], [0, 1], 'b--')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve for AdaBoost horse colic detection system')
    ax.axis([0, 1, 0, 1])
    plt.show()
    print("the Area Under the Curve is: ", ySum*xStep)






















