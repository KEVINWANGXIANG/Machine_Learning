from sklearn.svm import SVC, LinearSVC
import pandas
import numpy

'''
实验
X = [[0], [1], [2], [3]]
Y = [0, 1, 2, 3]
clf = SVC(decision_function_shape='ovo') #ovo为一对一,还有ovr
clf.fit(X, Y)

dec = clf.decision_function([[1]]) #返回的是样本距离超平面的距离
print(dec)

print("预测:", clf.predict([[5]]))
'''
#鸢尾花分类
def createTrainDataSet(path):
    fname = open(path)
    dataf = pandas.read_csv(fname)
    dataSet = dataf.iloc[:, 1:len(dataf.columns) - 1].as_matrix()
    dataSet = list(dataSet)
    b = []
    for i in range(len(dataSet)):
        b.append(list(dataSet[i]))
    labelSet = dataf.iloc[:, len(dataf.columns) - 1: len(dataf.columns)].as_matrix()
    labelSet = list(labelSet)
    label = []
    labels = []
    for i in range(len(labelSet)):
        label.append(list(labelSet[i]))
    for i in range(len(label)):
        labels.append(label[i][0])
    for i in range(len(labels)):
        if labels[i] == "setosa":
            labels[i] = 1
        elif labels[i] == "versicolor":
            labels[i] = -1
        else:
            labels[i] = 0

    return b, labels

path = r'C:/Users/Administrator/Desktop/数据集dataset/trainData.csv'
trainDataSet, labels = createTrainDataSet(path)
trainDataSet = numpy.array(trainDataSet)
# print(trainDataSet)
# print(labels)
clf = SVC(decision_function_shape='ovo') #ovo为一对一,还有ovr
clf.fit(trainDataSet, labels)

def createTestDataSet(path):
    fname = open(path)
    dataf = pandas.read_csv(fname)
    dataSet = dataf.iloc[:, 1:len(dataf.columns) - 1].as_matrix()
    dataSet = list(dataSet)
    b = []
    for i in range(len(dataSet)):
        b.append(list(dataSet[i]))
    labelSet = dataf.iloc[:, len(dataf.columns) - 1: len(dataf.columns)].as_matrix()
    labelSet = list(labelSet)
    label = []
    labels = []
    for i in range(len(labelSet)):
        label.append(list(labelSet[i]))
    for i in range(len(label)):
        labels.append(label[i][0])
    # print(labels)
    for i in range(len(labels)):
        if labels[i] == "setosa":
            labels[i] = 1
        elif labels[i] == "versicolor":
            labels[i] = -1
        else:
            labels[i] = 0
    return b, labels
path1 = r'C:/Users/Administrator/Desktop/数据集dataset/testData.csv'
testData, rightClassify = createTestDataSet(path1)
# print(testData)
# print(type(testData))
# print(rightClassify)
errorCount = 0
for i in range(len(testData)):
    predict_label = clf.predict([testData[i]])
    # print(predict_label)
    if predict_label[0] != rightClassify[i]:
        errorCount += 1
print("预测的错误率是：%.5f" %(float(errorCount) / len(testData)))
