from numpy import *

def loadDataSet():
    return [[1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5]]

#创建大小为1的所有候选项集的集合
def createC1(dataSet):
    C1 = []
    for transaction in dataSet:
        for item in transaction:
            if not [item] in C1:
                C1.append([item])
    C1.sort()
    #frozenset是指冰冻的集合，就是说他们是不可改变的，即用户不能修改他们
    return list(map(frozenset, C1))

#数据集扫描, D为数据集，Ck为候选项列表，minSupport为最小支持度
def scanD(D, Ck, minSupport):
    ssCnt = {}
    for tid in D:
        for can in Ck:
            if can.issubset(tid):
                if can not in ssCnt:
                    ssCnt[can] = 1
                else:
                    ssCnt[can] += 1
    numItems = float(len(D))
    retList = []
    supportData = {}
    for key in ssCnt:
        support = ssCnt[key] / numItems
        if support >= minSupport:
            retList.insert(0, key)
        supportData[key] = support
    return retList, supportData

dataSet = loadDataSet()
# C1 = createC1(dataSet)
# # print(C1)
# #构建集合表示的数据集D
# D = list(map(set, dataSet))
# # print(D)
# L1, suppData0 = scanD(D, C1, 0.5)
# print(L1)
# print(suppData0)

#频繁项集到候选项集，LK为频繁项集列表，K为候选项集元素的个数
def aprioriGen(Lk, k):
    retList = []
    lenLk = len(Lk)
    for i in range(lenLk):
        for j in range(i + 1, lenLk):
            L1 = list(Lk[i])[:k-2]
            L2 = list(Lk[j])[:k-2]
            L1.sort()
            L2.sort()
            #如果前k-2项相同时，将两个集合合并
            if L1 == L2:
                retList.append(Lk[i] | Lk[j])
    return retList

#主函数
def apriori(dataSet, minSupport=0.5):
    C1 = createC1(dataSet)
    D = list(map(set, dataSet))
    L1, supportData = scanD(D, C1, minSupport)
    L = [L1]
    k = 2
    while (len(L[k-2]) > 0):
        Ck = aprioriGen(L[k-2], k)
        #扫描数据集，从Ck得到Lk
        Lk, supK = scanD(D, Ck, minSupport)
        # if Lk == []:
        #     break
        supportData.update(supK)
        L.append(Lk)
        k += 1
    return L, supportData
'''
L, suppData = apriori(dataSet)
print(L)
print(suppData)
print(L[0])
print(L[1])
print(L[2])
print(L[3])

L, suppData = apriori(dataSet, minSupport=0.7)
print(L)
print(suppData)
'''
#L:[[frozenset({5}), frozenset({2}), frozenset({3}), frozenset({1})], [frozenset({2, 3}), frozenset({3, 5}), frozenset({2, 5}), frozenset({1, 3})], [frozenset({2, 3, 5})], []]
#主函数,参数分别为频繁项集列表
#一步一步分析
def generateRules(L, supportData, minConf=0.7):
    bigRuleList = []
    #只获取有两个或更多元素的集合
    for i in range(1, len(L)):
        for freqSet in L[i]:
            H1 = [frozenset([item]) for item in freqSet]
            # print("11111")
            # print(H1)
            if (i > 1):
                rulesFromConseq(freqSet, H1, supportData, bigRuleList, minConf)
            else:
                calcConf(freqSet, H1, supportData, bigRuleList, minConf)
    return bigRuleList

#规则评估
#参数为频繁项集，H为规则右部的列表
def calcConf(freqSet, H, supportData, br1, minConf=0.7):
    #保存满足最小可信度要求的规则列表
    prunedH = []
    for conseq in H:
        conf = supportData[freqSet] / supportData[freqSet - conseq]
        if conf >= minConf:
            print(freqSet - conseq, '-->', conseq, 'conf:', conf)
            br1.append((freqSet - conseq, conseq, conf))
            prunedH.append(conseq)
    return prunedH

#生成候选规则集合,H为可以出现在规则右部的元素列表
def rulesFromConseq(freqSet, H, supportData, br1, minConf=0.7):
    m = len(H[0])
    # print("2222")
    # print(H[0])
    if (len(freqSet) > (m + 1)):
        Hmp1 = aprioriGen(H, m+1)
        # print("3333")
        # print(Hmp1)
        Hmp1 = calcConf(freqSet, Hmp1, supportData, br1, minConf)
        # print("4444")
        # print(Hmp1)
        if (len(Hmp1) > 1):
            rulesFromConseq(freqSet, Hmp1, supportData, br1, minConf)


# L, suppData = apriori(dataSet, minSupport=0.5)
# # print(L)
# rules = generateRules(L, suppData, minConf=0.7)
# print(rules)
# rules = generateRules(L, suppData, minConf=0.5)
# print(rules)

#发现毒蘑菇的相似特征
mushDataSet = [line.split() for line in open(r'F:/Python/机器学习实战/MLiA_SourceCode/machinelearninginaction/Ch11/mushroom.dat').readlines()]
# print(mushDataSet)
# print(len(mushDataSet))
L, suppData = apriori(mushDataSet, minSupport=0.3)
# print(L)
#搜索包含有毒特征值2的频繁项集
# for item in L[1]:
#     if item.intersection('2'):
#         print(item)
for item in L[3]:
    if item.intersection('2'):
        print(item)















