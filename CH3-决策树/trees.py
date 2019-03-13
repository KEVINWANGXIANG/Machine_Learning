from math import log
import operator
import pickle
import treePlotter

#计算给定数据集的香浓熵
def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1] #获取每一条数据的类别
        #计算每个类别的个数
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt

def createDataSet():
    dataSet = [[1, 1, 'yes'], [1, 1, 'yes'], [1, 0, 'no'], [0, 1, 'no'], [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    # dataSet = [['青年', '否', '否', '一般', '否'],
    #            ['青年', '否', '否', '好', '否'],
    #            ['青年', '是', '否', '好', '是'],
    #            ['青年', '是', '是', '一般', '是'],
    #            ['青年', '否', '否', '一般', '否'],
    #            ['中年', '否', '否', '一般', '否'],
    #            ['中年', '否', '否', '好', '否'],
    #            ['中年', '是', '是', '好', '是'],
    #            ['中年', '否', '是', '非常好', '是'],
    #            ['中年', '否', '是', '非常好', '是'],
    #            ['老年', '否', '是', '非常好', '是'],
    #            ['老年', '否', '是', '好', '是'],
    #            ['老年', '是', '否', '好', '是'],
    #            ['老年', '是', '否', '非常好', '是'],
    #            ['老年', '否', '否', '一般', '否']]
    # labels = ['年龄', '有工作', '有自己的房子', '信贷情况', '类别']
    return dataSet, labels

myDat, labels = createDataSet()
# shannonEnt = calcShannonEnt(myDat)
# print(shannonEnt)
# myDat[0][-1] = 'maybe'
# # print(myDat)
# print(calcShannonEnt(myDat))

#按照给定特征划分数据集
#三个参数分别是：待划分的数据集、划分数据集的特征、需要返回的特征的值
def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet

# list = splitDataSet(myDat, 0, 1)
# print(list)

#选择最好的数据集划分方式
def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1
    baseEntropy = calcShannonEnt(dataSet) #计算数据集的经验熵
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet] #遍历每一特征列，即一列一列的
        uniqueVals = set(featList) #去重，算出这一特征的取值
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value) #划分数据集
            prob = len(subDataSet) / float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy
        if (infoGain > bestInfoGain):
            bestInfoGain = infoGain #最大的信息增益
            bestFeature = i #最好的特征编号
    return bestFeature


# print(chooseBestFeatureToSplit(myDat))
# print(myDat)

#当处理完了所有属性，类标签不是唯一的，采用多数表决的方法，方法如下：
def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

#创建树
def createTree(dataSet, labels):
    classList = [example[-1] for example in dataSet] #取数据集中最后一列，即标签列
    if classList.count(classList[0]) == len(classList): #如果所有类别完全相同，则停止划分，并返回该唯一的类
        return classList[0]
    if len(dataSet[0]) == 1: #如果遍历完所有特征标签仍然不是唯一的，则采取多数表决方式
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat] #获取最好的特征
    myTree = {bestFeatLabel: {}} #初始化树
    del(labels[bestFeat]) #删除刚刚最好的特征
    featValues = [example[bestFeat] for example in dataSet] #得到最好属性列表包含的所有属性值
    uniqueVals = set(featValues)#去重
    for value in uniqueVals:
        subLabels = labels[:] #为什么闲的蛋疼再赋一次值呢，因为为了保证每次调用函数createTree时不改变原始列表的内容，使用新变量代替原始列表
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)#递归调用产生子树
    return myTree

#生成树
myDat, labels = createDataSet()
myTree = createTree(myDat, labels)
# print(myTree)

def classify(inputTree, featLabels, testVec):
    firstStr = list(inputTree.keys())[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = classify(secondDict[key], featLabels, testVec)
            else:
                classLabel = secondDict[key]
    return classLabel

myDat, labels = createDataSet()
# print(labels)
# print(myTree)
label = classify(myTree, labels, [1, 1])
# print(label)

#决策树的存储，存储到文件中
def storeTree(inputTree, filename):
    fw = open(filename, "wb")
    pickle.dump(inputTree, fw, 0)
    fw.close()
#从保存的文件中读取决策树模型
def grabTree(filename):
    fr = open(filename, 'rb')
    return pickle.load(fr)

filename = r'F:/Python/机器学习实战/CH3-决策树/classifierStorage.txt'
storeTree(myTree, filename)
# print(grabTree(filename))

#测试隐形眼镜数据集的数据集
fr = open(r'F:/Python/机器学习实战/CH3-决策树/lenses.txt')
lenses = [inst.strip().split('\t') for inst in fr.readlines()]
lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']
lensesTree = createTree(lenses, lensesLabels)
print(lensesTree)

treePlotter.createPlot(lensesTree)

filename_glass = r'F:/Python/机器学习实战/CH3-决策树/classifier_glass.txt'
storeTree(lensesTree, filename_glass)
print(grabTree(filename_glass))
















