import numpy
import operator
import matplotlib
import matplotlib.pyplot as plt

plt.rc('font', family="SimHei", size=13) #设置字体

def createDataSet():
    group = numpy.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels

def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0] #获取数组的长度，即数据集的长度
    diffMat = numpy.tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1) #axis=1 按行相加 axis=0 按列相加
    distances = sqDistances ** 0.5
    sortedDistance = distances.argsort() #从大到小排列，提取对应的索引，并输出
    # print(sortedDistance)
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistance[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1 #存在则加一，不存在等于1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

dataSet, labels = createDataSet()
label = classify0([0.2, 0.2], dataSet, labels, 3)
# print(label)

def file2matrix(filename):
    fr = open(filename)
    arrayOLines = fr.readlines()
    numberOfLines = len(arrayOLines)
    returnMat = numpy.zeros((numberOfLines, 3))
    classLabelVector = []
    index = 0
    for line in arrayOLines:
        line = line.strip() #截取所有的回车符
        listFromLine = line.split('\t') #整行数据分割成一个元素列表
        returnMat[index, :] = listFromLine[0:3] #前三列提取作为数据
        classLabelVector.append(int(listFromLine[-1]))#获取这一行数据的标签
        index += 1
    return returnMat, classLabelVector #returnMat的类型是数组array，classLabelVector的类型是列表

filename = r'F:/Python/机器学习实战/MLiA_SourceCode/machinelearninginaction/Ch02/datingTestSet2.txt'
datingDataMat, datingLabels = file2matrix(filename)
# print(datingDataMat)
# print(datingLabels)

fig = plt.figure()
ax = fig.add_subplot(111) #111的含义是将图分为1行1列，此子图取第一个（从左向右从上往下数）
ax.scatter(datingDataMat[:, 0], datingDataMat[:, 1], 15.0*numpy.array(datingLabels), 15.0*numpy.array(datingLabels)) #利用变量存储的类标签属性绘制色彩不同的点
plt.xlabel("玩视频游戏所耗时间百分比")  #横坐标文字标签
plt.ylabel("每周消费的冰淇淋公升数")  #纵坐标文字标签
# plt.show()

#归一化特征值
def autoNorm(dataSet):
    minVals = dataSet.min(0) #每列的最小值，参数0使得函数从列中选取最小值，而不是选取当前行的最小值
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = numpy.zeros(numpy.shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - numpy.tile(minVals, (m, 1))
    normDataSet = normDataSet / numpy.tile(ranges, (m, 1))
    return normDataSet, ranges, minVals

normMat, ranges, minVals = autoNorm(datingDataMat)
# print(type(normMat))
# print(ranges)
# print(minVals)

#测试正确率
def datingClassTest():
    hoRatio = 0.1
    datingDataMat, datingLabels = file2matrix(r'F:/Python/机器学习实战/MLiA_SourceCode/machinelearninginaction/Ch02/datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio)
    errorCount = 0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i, :], normMat[0:m - numTestVecs, :], datingLabels[0:m - numTestVecs], 3)
        print("the classifier came back with: %d, the real answer is: %d" %(classifierResult, datingLabels[i]))
        if classifierResult != datingLabels[i]:
            errorCount += 1.0
    print("the total error rate is %.5f " % (errorCount / float(numTestVecs)))
#错误率约为0.02
# datingClassTest()

#由用户输入特征值，最后获取结果
def classifyPerson():
    resultList = ["not at all", "in small doses", "in large doses"]
    percentTats = float(input("percentage of time spent playing video game:"))
    ffMiles = float(input("frequent flier miles earned per year:"))
    iceCream = float(input("liters of ice cream consumed per year:"))
    datingDataMat, datingLabels = file2matrix(r'F:/Python/机器学习实战/MLiA_SourceCode/machinelearninginaction/Ch02/datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = numpy.array([ffMiles, percentTats,  iceCream])
    classifierResult = classify0((inArr - minVals) / ranges, normMat, datingLabels, 3)
    print("You will probably like this person:", resultList[classifierResult - 1])
#用户输入测试
# classifyPerson()



