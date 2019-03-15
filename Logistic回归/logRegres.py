import  math
import numpy
import matplotlib.pyplot as plt
import random

#加载训练数据集，返回训练数据和训练数据集的标签
def loadDataSet():
    dataMat = []
    labelMat = []
    fr = open(r'F:/Python/机器学习实战/MLiA_SourceCode/machinelearninginaction/Ch05/testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])]) #多加了一列
        labelMat.append(int(lineArr[2]))
    return dataMat, labelMat

#sigmoid函数
def sigmoid(inX):
    return 1.0 / (1 + numpy.exp(-inX))

#梯度上升法算法
def gradAscent(dataMatIn, classLabels):
    dataMatrix = numpy.mat(dataMatIn)
    labelMat = numpy.mat(classLabels).transpose() #转置
    m, n = numpy.shape(dataMatrix)
    alpha = 0.001
    maxCycles = 500#迭代次数
    weights = numpy.ones((n, 1))
    #迭代500次，无限逼近真实值
    for k in range(maxCycles):
        h = sigmoid(dataMatrix * weights)
        error = (labelMat - h)
        weights = weights + alpha * dataMatrix.transpose() * error
    return weights #返回回归系数

dataArr, labelMat = loadDataSet()
# print(dataArr)
weights = gradAscent(dataArr, labelMat)

#画出数据集和Logistic回归最佳拟合直线的函数
def plotBestFit(weights):
    dataMat, labelMat = loadDataSet()
    dataArr = numpy.array(dataMat)
    n = numpy.shape(dataArr)[0]
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i, 1])
            ycord1.append(dataArr[i, 2])
        else:
            xcord2.append(dataArr[i, 1])
            ycord2.append(dataArr[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='blue')
    x = numpy.arange(-3.0, 3.0, 0.1)#第一个参数为起点，第二个为终点，第三个为步长
    y = (-weights[0] - weights[1] * x) / weights[2]
    ax.plot(x, y)
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.show()

# plotBestFit(weights.getA())

#随机梯度上升算法
def stocGradAscent0(dataMatrix, classLabels):
    m,n = numpy.shape(dataMatrix)
    alpha = 0.01
    weights = numpy.ones(n)
    for i in range(m):
        h = sigmoid(sum(dataMatrix[i] * weights))
        error = classLabels[i] - h
        weights = weights + alpha * error * dataMatrix[i]
    return weights

# weights = stocGradAscent0(numpy.array(dataArr), labelMat)
# plotBestFit(weights)

#改进的随机梯度上升算法
def stocGradAscent1(dataMatrix, classLabels, numIter=150):
    m, n = numpy.shape(dataMatrix)
    weights = numpy.ones(n)
    for j in range(numIter):
        dataIndex = list(range(m))
        for i in range(m):
            alpha = 4 / (1.0 + j + i) + 0.01
            randIndex = int(random.uniform(0, len(dataIndex)))
            h = sigmoid(sum(dataMatrix[randIndex] * weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * dataMatrix[randIndex]
            del(dataIndex[randIndex])
    return weights

weights = stocGradAscent1(numpy.array(dataArr), labelMat)
# plotBestFit(weights)


#从疝气病症预测病马的死亡率
def classifyVector(inX, weights):
    prob = sigmoid(sum(inX * weights))
    if prob > 0.5:
        return 1.0
    else:
        return 0.0

def colicTest():
    frTrain = open(r'F:/Python/机器学习实战/MLiA_SourceCode/machinelearninginaction/Ch05/horseColicTraining.txt')
    frTest = open(r'F:/Python/机器学习实战/MLiA_SourceCode/machinelearninginaction/Ch05/horseColicTest.txt')
    trainingSet = []
    trainingLabels = []
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[21]))
    trainWeights = stocGradAscent1(numpy.array(trainingSet), trainingLabels, 500)
    errorCount = 0
    numTestVec = 0
    for line in frTest.readlines():
        numTestVec += 1.0
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(numpy.array(lineArr), trainWeights)) != int(currLine[21]):
            errorCount += 1
    errorRate = (float(errorCount) / numTestVec)
    print("the error rate of this test is %f" %errorRate)
    return errorRate

#测试十次，取平均值
def multiTest():
    numTests = 10
    errorSum = 0.0
    for k in range(numTests):
        errorSum += colicTest()
    print("after %d iterations the average rate is:%f" %(numTests, errorSum / float(numTests)))

multiTest()



























