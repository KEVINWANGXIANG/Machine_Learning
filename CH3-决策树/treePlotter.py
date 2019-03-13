import matplotlib.pyplot as plt

plt.rc('font', family="SimHei", size=13) #设置字体

decisionNode = dict(boxstyle="sawtooth", fc="0.8")# 决策节点的属性。boxstyle为文本框的类型，sawtooth是锯齿形，fc是边框线粗细
leafNode = dict(boxstyle="round4", fc="0.8")#决策树叶子节点的属性
arrow_args = dict(arrowstyle="<-")#箭头的属性

#绘制带箭头的注解
def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    createPlot.ax1.annotate(nodeTxt, xy=parentPt, xycoords='axes fraction',
    xytext=centerPt, textcoords='axes fraction', va="center", ha="center", bbox=nodeType, arrowprops=arrow_args)

# def createPlot():
#     fig = plt.figure(1, facecolor='white')
#     fig.clf()
#     createPlot.ax1 = plt.subplot(111, frameon=False)
#     plotNode('决策结点', (0.5, 0.1), (0.1, 0.5), decisionNode)
#     plotNode('叶节点', (0.8, 0.1), (0.3, 0.8), leafNode)
#     # plotNode('祥哥最帅', (1.0, 0.2), (0.3, 0.8), decisionNode)
#     plt.show()

# createPlot()

#获取叶节点的数目
def getNumLeafs(myTree):
    numLeafs = 0
    firstStr = list(myTree.keys())[0] #获取根节点
    secondDict = myTree[firstStr] #获取根节点的子节点
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == "dict":#判断是否是字典类型
            numLeafs += getNumLeafs(secondDict[key]) #递归调用分别求根节点的叶子节点个数
        else:
            numLeafs += 1 #如果不是字典类型，说明是叶子节点，叶子节点数加一
    return numLeafs

#获取树的层数
def getTreeDepth(myTree):
    maxDepth = 0
    firstStr = list(myTree.keys())[0] #获取根节点
    secondDict = myTree[firstStr] #获取根节点的子节点
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == "dict":
            thisDepth = 1 + getTreeDepth(secondDict[key]) #递归
        else:
            thisDepth = 1
        if thisDepth > maxDepth:
            maxDepth = thisDepth
    return maxDepth

def retrieveTree(i):
    listOfTrees = [{'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}}, {'no surfacing': {0: 'no', 1: {'flippers': {0: {'head': {0: 'no', 1: 'yes'}}, 1: 'no'}}}},
                   {'有自己的房子': {'是': '是', '否': {'有工作': {'是': '是', '否': '否'}}}}]
    return listOfTrees[i]

# print(retrieveTree(1))
myTree = retrieveTree(0)
#获取树的深度
depth = getTreeDepth(myTree)
print(depth)
#获取叶子结点数
numLeafs = getNumLeafs(myTree)
print(numLeafs)

#计算父节点和子节点的中间位置
def plotMidText(cntrPt, parentPt, txtString):
    xMid = (parentPt[0] - cntrPt[0]) / 2.0 + cntrPt[0]
    yMid = (parentPt[1] - cntrPt[1]) / 2.0 + cntrPt[1]
    createPlot.ax1.text(xMid, yMid, txtString)

def plotTree(myTree, parentPt, nodeTxt):
    numLeafs = getNumLeafs(myTree)
    depth = getTreeDepth(myTree)
    firstStr = list(myTree.keys())[0]
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs)) / 2.0 / plotTree.totalW, plotTree.yOff)
    plotMidText(cntrPt, parentPt, nodeTxt)
    plotNode(firstStr, cntrPt, parentPt, decisionNode)
    secondDict = myTree[firstStr]
    plotTree.yOff = plotTree.yOff - 1.0/plotTree.totalD
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            plotTree(secondDict[key], cntrPt, str(key))
        else:
            plotTree.xOff = plotTree.xOff + 1.0 / plotTree.totalW
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
    plotTree.yOff = plotTree.yOff + 1.0 / plotTree.totalD

def createPlot(inTree):
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)
    plotTree.totalW = float(getNumLeafs(inTree))
    plotTree.totalD = float(getTreeDepth(inTree))
    plotTree.xOff = -0.5 / plotTree.totalW
    plotTree.yOff = 1.0
    plotTree(inTree, (0.5, 1.0), '')
    plt.show()

createPlot(myTree)

















