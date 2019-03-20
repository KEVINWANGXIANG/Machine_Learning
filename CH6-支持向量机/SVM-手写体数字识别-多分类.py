from sklearn.svm import SVC, LinearSVC
import pandas
import numpy
import os

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

def trainData():
    labels = []
    trainFile = os.listdir(r'C:/Users/Administrator/Desktop/digits/digits/traindata')
    num = len(trainFile)
    trainArr = numpy.zeros((num, 1024))
    for i in range(0, num):
        thisfName = trainFile[i]
        thisLabel = sepLabel(thisfName)
        labels.append(thisLabel)
        trainArr[i, :] = dataToArray('C:/Users/Administrator/Desktop/digits/digits/traindata/' + thisfName)
    return trainArr, labels
trainData, labelsAll = trainData()
# print(labelsAll)
trainData = numpy.array(trainData)
clf = SVC(decision_function_shape='ovo')
print(clf.fit(trainData, labelsAll))

import time
#批量测试数字
startTime = time.time()
allLabels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
testList = os.listdir(r'C:/Users/Administrator/Desktop/digits/digits/testdata')
tNum = len(testList)
right_num = 0
num = 1
for i in range(0, tNum):
    thisTestFile = testList[i]
    true_label = sepLabel(thisTestFile)
    testArr = dataToArray('C:/Users/Administrator/Desktop/digits/digits/testdata/' + thisTestFile)
    thisLabel = clf.predict([testArr])
    print("第%d个文件的正确类别是:%d" % (num, true_label))
    print("第%d个文件的分类类别是:%d" % (num, thisLabel[0]))
    num += 1
    if thisLabel[0] == true_label:
        right_num += 1
rate = float(right_num / (num - 1))
print("识别的准确率是：%.5f" %(rate))
endTime = time.time()
print("总耗时为:%.2f" % (endTime - startTime))

#SVM算法太屌了,9秒完成，准确率97.3%