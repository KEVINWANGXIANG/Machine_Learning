from numpy import *

#FP树的类定义,nameValue是节点名称， numOccur为计数值
class treeNode:
    def __init__(self, nameValue, numOccur, parentNode):
        self.name = nameValue
        self.count = numOccur
        self.nodeLink = None
        self.parent = parentNode
        self.children = {}

    #对count变量增加给定值
    def inc(self, numOccur):
        self.count += numOccur
    #将树以文本形式显示,'  '*ind表示前面的空格数
    def disp(self, ind=1):
        print('  '*ind, self.name, ' ', self.count)
        for child in self.children.values():
            child.disp(ind+1)

# rootNode = treeNode('pyramid', 9, None)
# # rootNode.disp()
# rootNode.children['eye'] = treeNode('eye', 13, None)
# rootNode.children['phoenix'] = treeNode('phoenix', 3, None)
# # rootNode.disp()

#FP树构建函数
def createTree(dataSet, minSup=1):
    #头指针表
    headerTable = {}
    for trans in dataSet:
        for item in trans:
            #get(item, 0)表示的是如果存在item了，则返回item，否则返回0， dataSet[trans]为1，就是统计所有数据中单个字母出现的次数
            headerTable[item] = headerTable.get(item, 0) + dataSet[trans]
    #移除不满足最小支持度的元素项
    for k in list(headerTable.keys()):
        if headerTable[k] < minSup:
            del (headerTable[k])
    #字母的频繁项集
    freqItemSet = set(headerTable.keys())
    if len(freqItemSet) == 0:
        return None, None
    #多开辟一个位置，用来存放头指针
    for k in headerTable:
        headerTable[k] = [headerTable[k], None]
    #创建头结点
    retTree = treeNode('Null Set', 1, None)
    #将每一条数据插进FP树中，期间需要去除数据中出现次数低于阈值的字母，且数据中字母按出现次数进行降序排序
    for tranSet, count in dataSet.items():
        #用来存放该数据中符合条件的字母
        localD = {}
        for item in tranSet:
            if item in freqItemSet:
                localD[item] = headerTable[item][0]
        #降序排序,并插入到树中
        if len(localD) > 0:
            orderedItems = [v[0] for v in sorted(localD.items(), key=lambda p:p[1], reverse=True)]
            #插入到FP树中
            updateTree(orderedItems, retTree, headerTable, count)
    return retTree, headerTable

#将一条数据插进FP树中
def updateTree(items, inTree, headerTable, count):
    #如果首字母的节点存在，则直接更新节点的计数器
    if items[0] in inTree.children:
        inTree.children[items[0]].inc(count)
    else:
        #创建新节点，之后需要将该节点放进字母表的链表中
        inTree.children[items[0]] = treeNode(items[0], count, inTree)
        #如果该字母首次出现，则直接将该字母的头指针指向该节点
        if headerTable[items[0]][1] == None:
            headerTable[items[0]][1] = inTree.children[items[0]]
        else:
            #否则，需要将其插入到合适的位置，本文采用的是尾插法
            updateHeader(headerTable[items[0]][1], inTree.children[items[0]])
    if len(items) > 1:
        updateTree(items[1::], inTree.children[items[0]], headerTable, count)

#将新建的字母节点加入到字母链表的链尾， 尾插法
def updateHeader(nodeToTest, targetNode):
    while (nodeToTest.nodeLink != None):
        nodeToTest = nodeToTest.nodeLink
    nodeToTest.nodeLink = targetNode

def loadSimpleDat():
    simpDat = [['r', 'z', 'h', 'j', 'p'],
               ['z', 'y', 'x', 'w', 'v', 'u', 't', 's'],
               ['z'],
               ['r', 'x', 'n', 'o', 's'],
               ['y', 'r', 'x', 'z', 'q', 't', 'p'],
               ['y', 'z', 'x', 'e', 'q', 's', 't', 'm']]
    return simpDat

def createInitSet(dataSet):
    retDict = {}
    for trans in dataSet:
        retDict[frozenset(trans)] = 1
    return retDict

# simpleDat = loadSimpleDat()
# # print(simpleDat)
# initSet = createInitSet(simpleDat)
# # print(initSet)
# myFPtree, myHeaderTab = createTree(initSet, 3)
# myFPtree.disp()
# print(myHeaderTab)
# print(myHeaderTab['x'][1])

#在FP树中，从一个节点开始，上溯至根节点，并记录路径，这样就找到了频繁项集的一个前缀路径
def ascendTree(leafNode, prefixPath):
    if leafNode.parent != None:
        prefixPath.append(leafNode.name)
        ascendTree(leafNode.parent, prefixPath)

#在FP树中，找出某个字母所有的前缀路径，即找到对应的条件模式基、
def findPrefixPath(basePat, treeNode):
    #存储前缀路径，用字典因为要记录每条前缀路径的出现次数
    conPats = {}
    while treeNode != None:
        prefixPath = []
        ascendTree(treeNode, prefixPath)
        if len(prefixPath) > 1:
            conPats[frozenset(prefixPath[1:])] = treeNode.count
        treeNode = treeNode.nodeLink
    return conPats
# path1 = findPrefixPath('t', myHeaderTab['t'][1])
# print(path1)
#
# path2 = findPrefixPath('z', myHeaderTab['z'][1])
# print(path2)
#
# path3 = findPrefixPath('r', myHeaderTab['r'][1])
# print(path3)

#递归地从FP树中挖掘频繁项集
#preFix当前频繁项集合的前缀，freqItemList存储频繁项集
def mineTree(inTree, headerTable, minSup, preFix, freqItemList):
    bigL = [v[0] for v in sorted(headerTable.items(), key=lambda p: p[1][0])]  # 对字母表进行排序（根据出现次数）
    #枚举字母表中的每一个字母
    for basePat in bigL:
        newFreqSet = preFix.copy()
        #将该字母加入到前缀中，形成新的频繁项集
        newFreqSet.add(basePat)
        #保存新的频繁项集
        freqItemList.append(newFreqSet)
        #在当前FP树中找到该字母的条件模式基
        condPattBases = findPrefixPath(basePat, headerTable[basePat][1])
        #利用条件模式基创建新的FP树
        myCondTree, myHead = createTree(condPattBases, minSup)
        #如果裁剪后的FP树仍不为空，则将新的频繁项集作为当前频繁项集的前缀，然后在新的FP树上挖掘频繁项集
        if myHead != None:
            print("conditional tree for:", newFreqSet)
            myCondTree.disp(1)
            mineTree(myCondTree, myHead, minSup, newFreqSet, freqItemList)

freqItems = []
# mineTree(myFPtree, myHeaderTab, 3, set([]), freqItems)
# print(freqItems)
parsedDat = [line.split() for line in open('F:/Python/机器学习实战/MLiA_SourceCode/machinelearninginaction/Ch12/kosarak.dat').readlines()]
initSet = createInitSet(parsedDat)
myFPtree, myHeaderTab = createTree(initSet, 100000)
myFreqList = []
mineTree(myFPtree, myHeaderTab, 100000, set([]), myFreqList)
print(myFreqList)









