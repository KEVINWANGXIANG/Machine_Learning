from numpy import *

def loadDataSet(fileName):
    dataMat = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        #将每行映射成浮点数
        fltLine = list(map(float, curLine))
        dataMat.append(fltLine)
    return dataMat

#二分数据集
def binSplitDataSet(dataSet, feature, value):
    mat0 = dataSet[nonzero(dataSet[:, feature] > value)[0], :]
    mat1 = dataSet[nonzero(dataSet[:, feature] <= value)[0], :]
    return mat0, mat1

# testMat = mat(eye(4))
# print(testMat)
# mat0, mat1 = binSplitDataSet(testMat, 1, 0.5)
# print(mat0)
# print(mat1)

#创建叶节点的函数
def regLeaf(dataSet):
    return mean(dataSet[:, -1])

#计算误差
def regErr(dataSet):
    return var(dataSet[:, -1]) * shape(dataSet)[0]

#选择最好的切分特征和特征值
def chooseBestSplit(dataSet, leafType=regLeaf, errType=regErr, ops=(1, 4)):
    #容许的误差下降值
    tolS = ops[0]
    #切分的最少样本数
    tolN = ops[1]
    #如果结果只有一种，则直接生成叶节点
    if len(set(dataSet[:, -1].T.tolist()[0])) == 1:
        return None, leafType(dataSet)
    m, n = shape(dataSet)
    #计算整个数据的误差
    S = errType(dataSet)
    bestS = inf
    bestIndex = 0
    bestValue = 0
    for featIndex in range(n - 1):
        for splitVal in set((dataSet[:, featIndex].T.A.tolist())[0]):
            mat0, mat1 = binSplitDataSet(dataSet, featIndex, splitVal)
            if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN):
                continue
            newS = errType(mat0) + errType(mat1)
            if newS < bestS:
                bestIndex = featIndex
                bestValue = splitVal
                bestS = newS
    #如果误差减少不大则退出
    if (S - bestS) < tolS:
        return None, leafType(dataSet)
    mat0, mat1 = binSplitDataSet(dataSet, bestIndex, bestValue)
    #如果切分出的数据集很小则退出
    if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN):
        return None, leafType(dataSet)
    return bestIndex, bestValue


#创建树
def createTree(dataSet, leafType=regLeaf, errType=regErr, ops=(1, 4)):
    feat, val = chooseBestSplit(dataSet, leafType, errType, ops)
    if feat == None:
        return val
    retTree = {}
    retTree['spInd'] = feat
    retTree['spVal'] = val
    lSet, rSet = binSplitDataSet(dataSet, feat, val)
    retTree['left'] = createTree(lSet, leafType, errType, ops)
    retTree['right'] = createTree(rSet, leafType, errType, ops)
    return retTree

import matplotlib.pyplot as plt
# myDat2 = loadDataSet(r'F:/Python/机器学习实战/MLiA_SourceCode/machinelearninginaction/Ch09/ex0.txt')
# myMat2 = mat(myDat2)
# print(myMat)
# retTree = createTree(myMat)
# print(retTree)
# plt.plot(myMat[:, 1], myMat[:, 2], 'ro')
# plt.show()

# myDat = loadDataSet(r'F:/Python/机器学习实战/MLiA_SourceCode/machinelearninginaction/Ch09/ex2.txt')
# myMat = mat(myDat)
# print(myMat)
# retTree = createTree(myMat)
# print(retTree)

#判断当前处理的节点是否是叶节点
def isTree(obj):
    return (type(obj).__name__ == 'dict')

#递归函数，从上而下遍历直到叶节点为止，如果找到两个叶节点计算他们的平均值
def getMean(tree):
    if isTree(tree['right']):
        tree['right'] = getMean(tree['right'])
    if isTree(tree['left']):
        tree['left'] = getMean(tree['left'])
    return (tree['right'] + tree['left']) / 2

#剪枝
def prune(tree, testData):
    if shape(testData)[0] == 0:
        return getMean(tree)
    if (isTree(tree['right']) or isTree(tree['left'])):
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
    if isTree(tree['left']):
        tree['left'] = prune(tree['left'], lSet)
    if isTree(tree['right']):
        tree['right'] = prune(tree['right'], rSet)
    if not isTree(tree['left']) and not isTree(tree['right']):
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
        errorNoMerge = sum(power(lSet[:, -1] - tree['left'], 2)) + sum(power(rSet[:, -1] - tree['right'], 2))
        treeMean = (tree['left'] + tree['right']) / 2
        errorMerge = sum(power(testData[:, -1] - treeMean, 2))
        if errorMerge < errorNoMerge:
            print("merging")
            return treeMean
        else:
            return tree
    else:
        return tree


# myDataTest = loadDataSet(r'F:/Python/机器学习实战/MLiA_SourceCode/machinelearninginaction/Ch09/ex2test.txt')
# myTree = createTree(myMat2, ops=(0, 1))
# myMat2Test = mat(myDataTest)
# tree = prune(myTree, myMat2Test)
# print(tree)

#将数据集格式变成目标变量Y和自变量X
def linearSolve(dataSet):
    m, n = shape(dataSet)
    X = mat(ones((m, n)))
    Y = mat(ones((m, 1)))
    X[:, 1:n] = dataSet[:, 0:n-1]
    Y = dataSet[:, -1]
    xTx = X.T * X
    if linalg.det(xTx) == 0.0:
        raise NameError("this matrix is singular, cannot do inverse, try increasing the second value of ops")
    ws = xTx.I * (X.T * Y)
    return ws, X, Y

#生成叶节点的模型
def modelLeaf(dataSet):
    ws, X, Y = linearSolve(dataSet)
    return ws

#计算误差
def modelErr(dataSet):
    ws, X, Y = linearSolve(dataSet)
    yHat = X * ws
    return sum(power(Y - yHat, 2))

# myMat2 = mat(loadDataSet(r'F:/Python/机器学习实战/MLiA_SourceCode/machinelearninginaction/Ch09/exp2.txt'))
# myTree = createTree(myMat2, modelLeaf, modelErr, (1, 10))
# print(myTree)

#回归树叶节点构造
def regTreeEval(model, inDat):
    return float(model)
#模型树叶节点的构造
def modelTreeEval(model, inDat):
    n = shape(inDat)[1]
    X = mat(ones((1, n + 1)))
    X[:, 1:n+1] = inDat
    return float(X * model)

def treeForestCast(tree, inData, modelEval=regTreeEval):
    if not isTree(tree):
        return modelEval(tree, inData)
    if inData[tree['spInd']] > tree['spVal']:
        if isTree(tree['left']):
            return treeForestCast(tree['left'], inData, modelEval)
        else:
            return modelEval(tree['left'], inData)
    else:
        if isTree(tree['right']):
            return treeForestCast(tree['right'], inData, modelEval)
        else:
            return modelEval(tree['right'], inData)
#多次调用上一个函数
def createForeCast(tree, testData, modelEval=regTreeEval):
    m = len(testData)
    yHat = mat(zeros((m, 1)))
    for i in range(m):
        yHat[i, 0] = treeForestCast(tree, mat(testData[i]), modelEval)
    return yHat

# trainMat = loadDataSet(r'F:/Python/机器学习实战/MLiA_SourceCode/machinelearninginaction/Ch09/bikeSpeedVsIq_train.txt')
# testMat = loadDataSet(r'F:/Python/机器学习实战/MLiA_SourceCode/machinelearninginaction/Ch09/bikeSpeedVsIq_test.txt')
# trainMat = mat(trainMat)
# testMat = mat(testMat)
# #回归树
# myTree = createTree(trainMat, ops=(1, 20))
# yHat = createForeCast(myTree, testMat[:, 0])
# corr_regTree = corrcoef(yHat, testMat[:, 1], rowvar=0)[0, 1]
# print(corr_regTree)
# #m模型树
# myTree = createTree(trainMat, modelLeaf, modelErr, (1, 20))
# yHat = createForeCast(myTree, testMat[:, 0], modelTreeEval)
# corr_modelTree = corrcoef(yHat, testMat[:, 1], rowvar=0)[0, 1]
# print(corr_modelTree)
#
# ws, X, Y = linearSolve(trainMat)
# # print(ws)
# for i in range(shape(testMat)[0]):
#     yHat[i] = testMat[i, 0] * ws[1, 0] + ws[0, 0]
# corr_linear = corrcoef(yHat, testMat[:, 1], rowvar=0)[0, 1]
# print(corr_linear)


# from tkinter import *
# root = Tk()
# root.geometry("500x500")
#
# myLabel = Label(root, text="hello world")
# myLabel.grid()
#
#
#
#
# root.mainloop()












