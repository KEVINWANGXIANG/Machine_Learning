import numpy
from numpy import *
import time
import os
from PIL import Image
import operator
def knn(k, testData, trainData, labels):
    #获取数组的行数
    trainDataSize = trainData.shape[0]
    #扩展并计算对应差值
    dif = numpy.tile(testData, (trainDataSize, 1)) - trainData
    sqdif = dif ** 2
    #axis=1表示按行相加，axis=0表示按列相加
    sumSqdf = sqdif.sum(axis=1)
    distance = sumSqdf ** 0.5
    sortDistance = distance.argsort()
    # print(sortDistance)
    count = {}
    for i in range(0, k):
        vote = labels[sortDistance[i]]
        count[vote] = count.get(vote, 0) + 1
    sortCount = sorted(count.items(), key=operator.itemgetter(1), reverse=True)
    # print(sortCount)
    return sortCount[0][0]

#建立一个函数取文件名的前缀
def sepLabel(fname):
    fileStr = fname.split(".")[0]
    label = fileStr.split("_")[0]
    label = int(label)
    return label

#加载数据
def dataToArray(fname):
    arr = []
    fh = open(fname)
    for i in range(0, 32):
        thisline = fh.readline()
        for j in range(0, 32):
            arr.append(int(thisline[j]))
    return arr

# arr1 = dataToArray(r'C:/Users/Administrator/Desktop/digits/digits/traindata/0_0.txt')
# print(arr1)
#建立训练数据集
def trainData():
    labels = []
    trainFile = os.listdir(r'F:/Python/机器学习实战/MLiA_SourceCode/machinelearninginaction/Ch02/digits/trainingDigits')
    num = len(trainFile)
    trainArr = numpy.zeros((num, 1024))
    for i in range(0, num):
        thisfName = trainFile[i]
        thisLabel = sepLabel(thisfName)
        labels.append(thisLabel)
        trainArr[i, :] = dataToArray('F:/Python/机器学习实战/MLiA_SourceCode/machinelearninginaction/Ch02/digits/trainingDigits/' + thisfName)
    return trainArr, labels

def dataTest():
    trainArr, labels = trainData()
    testList = os.listdir(r'F:/Python/机器学习实战/MLiA_SourceCode/machinelearninginaction/Ch02/digits/testDigits')
    tNum = len(testList)
    right_num = 0
    num = 1
    for i in range(0, tNum):
        thisTestFile = testList[i]
        true_label = sepLabel(thisTestFile)
        testArr = dataToArray('F:/Python/机器学习实战/MLiA_SourceCode/machinelearninginaction/Ch02/digits/testDigits/' + thisTestFile)
        rknn = knn(3, testArr, trainArr, labels)
        print("第%d个文件的正确类别是:%d" % (num, true_label))
        print("第%d个文件的分类类别是:%d" % (num, rknn))
        num += 1
        if rknn == true_label:
            right_num += 1
    rate = float(right_num / (num - 1))
    print("识别的准确率是：%.5f" %(rate))
# startTime = time.time()
# dataTest()
# endTime = time.time()
# print("总耗时为%.2f" % (endTime - startTime))

#识别手写体图片
def pic2txt(imgPath, txtPath):
    im = Image.open(imgPath)
    fh = open(txtPath, "a")
    for i in range(0, 32):
        for j in range(0, 32):
            cl = im.getpixel((j, i))
            clAll = cl[0] + cl[1] + cl[2]
            if clAll == 0:
                fh.write("1")
            else:
                fh.write("0")
        fh.write("\n")
    fh.close()

#单个文件测试
def singleTest(file):
    trainArr, labels = trainData()
    testArr = dataToArray(file)
    rknn = knn(3, testArr, trainArr, labels)
    print(rknn)

txtPath = r'C:/Users/Administrator/Desktop/num.txt'
pic2txt(r'C:/Users/Administrator/Desktop/test.png', txtPath)
singleTest(r'C:/Users/Administrator/Desktop/num.txt')

