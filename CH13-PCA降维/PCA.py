
from numpy import *

def loadDataSet(fileName, delim='\t'):
    fr = open(fileName)
    stringArr = [line.strip().split(delim) for line in fr.readlines()]
    datArr = [list(map(float, line)) for line in stringArr]
    return mat(datArr)

#topNfeat为应用的N个特征
def pca(dataMat, topNfeat=999999):
    meanVals = mean(dataMat, axis=0)
    meanRemoved = dataMat - meanVals
    #协方差矩阵
    covMat = cov(meanRemoved, rowvar=0)
    #特征值和特征向量
    eigVals, eigVects = linalg.eig(mat(covMat))
    #获取从小到大排序特征值的下标
    eigValInd = argsort(eigVals)
    #获取最大特征值的下标N个（起始位置：结束位置：步长）
    eigValInd = eigValInd[:-(topNfeat + 1):-1]
    #最大特征值对应的特征向量
    redEigVects = eigVects[:, eigValInd]
    #将数据转换到新空间（特征向量*数据集）
    lowDDataMat = meanRemoved * redEigVects
    #将降维之后的数据映射到原空间，作用：用于测试，可以和未压缩的原始矩阵比较
    reconMat = (lowDDataMat * redEigVects.T) + meanVals
    #返回压缩后的矩阵和反构出的原始矩阵
    return lowDDataMat, reconMat

dataMat = loadDataSet(r'F:/Python/机器学习实战/MLiA_SourceCode/machinelearninginaction/Ch13/testSet.txt')
# print(dataMat)
# print(shape(dataMat))
# lowDMat, reconMat = pca(dataMat, 1)
# print(shape(lowDMat))
# print(reconMat)

# print(lowDMat)
import matplotlib
import matplotlib.pyplot as plt

# fig = plt.figure()
# ax = fig.add_subplot(111)
# # ax.scatter(dataMat[:, 0].flatten().A[0], dataMat[:, 1].flatten().A[0], marker='^', s=90)
# # ax.scatter(reconMat[:, 0].flatten().A[0], reconMat[:, 1].flatten().A[0], marker='o', s=50, c='red')
# plt.plot(lowDMat[:, 0], 'o')
# plt.show()

#将NaN替换成平均值的函数
def replaceNanWithMean():
    datMat = loadDataSet(r'F:/Python/机器学习实战/MLiA_SourceCode/machinelearninginaction/Ch13/secom.data', ' ')
    numFeat = shape(datMat)[1]
    for i in range(numFeat):
        meanVal = mean(datMat[nonzero(~isnan(datMat[:, i].A))[0], i])
        datMat[nonzero(isnan(datMat[:, i].A))[0], i] = meanVal
    return datMat
dataMat = replaceNanWithMean()
meanVals = mean(dataMat, axis=0)
meanRemoved = dataMat - meanVals
covMat = cov(meanRemoved, rowvar=0)
eigVals, eigVects = linalg.eig(mat(covMat))
# print(eigVals)
# print(len(eigVals))
# lowDMat, reconMat = pca(dataMat, 6)
# print(lowDMat)
eigValInd = argsort(eigVals)            #sort, sort goes smallest to largest
eigValInd = eigValInd[::-1]#reverse
sortedEigVals = eigVals[eigValInd]
total = sum(sortedEigVals)
varPercentage = sortedEigVals/total*100

#解决中文乱码问题
import matplotlib as mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['font.serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(range(1, 21), varPercentage[:20], marker='^')
plt.xlabel('主成分数目')
plt.ylabel('方差的百分比')
plt.show()

















