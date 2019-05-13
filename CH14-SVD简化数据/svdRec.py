from numpy import *

# U, Sigma, VT = linalg.svd([[1, 1], [7, 7]])
# # print(U)
# # print(Sigma)
# print(VT)

def loadExData():
    return [[1, 1, 1, 0, 0],
            [2, 2, 2, 0, 0],
            [1, 1, 1, 0, 0],
            [5, 5, 5, 0, 0],
            [1, 1, 0, 2, 2],
            [0, 0, 0, 3, 3],
            [0, 0, 0, 1, 1]]
# Data = loadExData()
# U, Sigma, VT = linalg.svd(Data)
# # print(Sigma)
# Sig3 = mat([[Sigma[0], 0, 0], [0, Sigma[1], 0], [0, 0, Sigma[2]]])
# #重构原始矩阵的近似矩阵
# RefactorVect = U[:, :3] * Sig3 * VT[:3, :]
# print(RefactorVect)

from numpy import linalg as la

#计算欧式距离
def euclidSim(inA, inB):
    return 1.0 / (1.0 + la.norm(inA - inB))

#计算皮尔逊相关系数
def pearsSim(inA, inB):
    if len(inA) < 3:
        return 1.0
    return 0.5 + 0.5 * corrcoef(inA, inB, rowvar=0)[0][1]

#计算余弦相似度
def cosSim(inA, inB):
    num = float(inA.T * inB)
    denom = la.norm(inA) * la.norm(inB)
    return 0.5 + 0.5 * (num / denom)

myMat = mat(loadExData())
# distance_eclu = euclidSim(myMat[:, 0], myMat[:, 4])
# print(distance_eclu)
# distance_eclu = euclidSim(myMat[:, 0], myMat[:, 0])
# print(distance_eclu)

# distance_cos = cosSim(myMat[:, 0], myMat[:, 4])
# print(distance_cos)
# distance_cos = cosSim(myMat[:, 0], myMat[:, 0])
# print(distance_cos)

# distance_pear = pearsSim(myMat[:, 0], myMat[:, 4])
# print(distance_pear)
# distance_pear = pearsSim(myMat[:, 0], myMat[:, 0])
# print(distance_pear)


#对物品评分
def standEst(dataMat, user, simMeas, item):
    n = shape(dataMat)[1]
    #对两个计算估计评分制变量初始化
    simTotal = 0.0
    ratSimTotal = 0.0
    for j in range(n):
        userRating = dataMat[user, j]
        if userRating == 0:
            continue
        #获取相比较的两列同时不为0的数据行号
        overLap = nonzero(logical_and(dataMat[:, item].A > 0, dataMat[:, j].A > 0))[0]
        # print(overLap)
        if len(overLap) == 0:
            similarity = 0
        else:
            similarity = simMeas(dataMat[overLap, item], dataMat[overLap, j])
        simTotal += similarity
        ratSimTotal += simTotal * userRating
    if simTotal == 0:
        return 0
    else:
        return ratSimTotal / simTotal

#产生最高的N个结果
def recommend(dataMat, user, N=3, simMeas=cosSim, estMethod=standEst):
    #寻找该用户未评级的物品编号
    unratedItems = nonzero(dataMat[user, :].A == 0)[1]
    if len(unratedItems) == 0:
        return "you rated everything"
    itemScores = []
    for item in unratedItems:
        estimatedScore = estMethod(dataMat, user, simMeas, item)
        itemScores.append((item, estimatedScore))
    #从大到小排列，取前N个
    return sorted(itemScores, key=lambda jj: jj[1], reverse=True)[:N]

# myMat = mat(loadExData())
# myMat[0, 1] = myMat[0, 0] = myMat[1, 0] = myMat[2, 0] = 4
# print(myMat)
myMat = mat([[4, 4, 0, 2, 2], [4, 0, 0, 3, 3], [4, 0, 0, 1, 1], [1, 1, 1, 2, 0], [2, 2, 2, 0, 0], [1, 1, 1, 0, 0], [5, 5, 5, 0, 0]])
# scores = recommend(myMat, 2)
# print(scores)

# scores = recommend(myMat, 2, simMeas=euclidSim)
# print(scores)

# scores = recommend(myMat, 2, simMeas=pearsSim)
# print(scores)

def loadExData2():
    return[[0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 5],
           [0, 0, 0, 3, 0, 4, 0, 0, 0, 0, 3],
           [0, 0, 0, 0, 4, 0, 0, 1, 0, 4, 0],
           [3, 3, 4, 0, 0, 0, 0, 2, 2, 0, 0],
           [5, 4, 5, 0, 0, 0, 0, 5, 5, 0, 0],
           [0, 0, 0, 0, 5, 0, 1, 0, 0, 5, 0],
           [4, 3, 4, 0, 0, 0, 0, 5, 5, 0, 1],
           [0, 0, 0, 4, 0, 4, 0, 0, 0, 0, 4],
           [0, 0, 0, 2, 0, 2, 5, 0, 0, 1, 2],
           [0, 0, 0, 0, 5, 0, 0, 0, 0, 4, 0],
           [1, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0]]


U, Sigma, VT = la.svd(mat(loadExData2()))
# print(Sigma)
# Sig2 = Sigma ** 2
# print(sum(Sig2))
# print(sum(Sig2) * 0.9)
# print(sum(Sig2[:3]))

#基于SVD的评分估计
def svdEst(dataMat, user, simMeas, item):
    n = shape(dataMat)[1]
    simTotal = 0.0
    ratSimTotal = 0.0
    U, Sigma, VT = la.svd(dataMat)
    #建立对角矩阵
    Sig4 = mat(eye(4) * Sigma[:4])
    #构建转换后的物品
    xformedItems = dataMat.T * U[:, :4] * Sig4.I
    for j in range(n):
        userRating = dataMat[user, j]
        if userRating == 0 or j == item:
            continue
        similarity = simMeas(xformedItems[item, :].T, xformedItems[j,:].T)
        print("the %d and %d similarity is: %f" %(item, j, similarity))
        simTotal += similarity
        ratSimTotal += similarity * userRating
    if simTotal == 0:
        return 0
    else:
        return ratSimTotal / simTotal
myMat = mat(loadExData2())
# scores = recommend(myMat, 1, 3, estMethod=svdEst)
# print(scores)

#基于SVD的图像压缩
#打印矩阵
def printMat(inMat, thresh=0.8):
    for i in range(32):
        for k in range(32):
            if float(inMat[i, k]) > thresh:
                print(1, end="")
            else:
                print(0, end="")
        print(' ')
#图像压缩
def imgCompress(numSV=3, thresh=0.8):
    myl = []
    for line in open(r'F:/Python/机器学习实战/MLiA_SourceCode/machinelearninginaction/Ch14/0_5.txt').readlines():
        newRow = []
        for i in range(32):
            newRow.append(int(line[i]))
        myl.append(newRow)
    myMat = mat(myl)
    print("****original matrix****")
    printMat(myMat, thresh)
    U, Sigma, VT = la.svd(myMat)
    #构建对角矩阵
    SigRecon = mat(zeros((numSV, numSV)))
    for k in range(numSV):
        SigRecon[k, k] = Sigma[k]
    reconMat = U[:, :numSV] * SigRecon * VT[:numSV, :]
    print("****reconstructed matrix using %d singular values" %(numSV))
    printMat(reconMat, thresh)

imgCompress(2)



















































