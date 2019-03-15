import numpy
import math
import random
import re
import feedparser

def loadDataSet():
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0, 1, 0, 1, 0, 1]
    return postingList, classVec

def createVocabList(dataSet):
    vocabSet = set([])  #创建一个空集
    for document in dataSet:
        vocabSet = vocabSet | set(document)#创建两个集合的并集
    return list(vocabSet)

#将词汇表转为向量
def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print("the word: %s is not in my Vocabulary!" %(word))
    return returnVec

listOPosts, listClasses = loadDataSet()
myVocabList = createVocabList(listOPosts)
# print(myVocabList)
# print(setOfWords2Vec(myVocabList, listOPosts[0]))
# print(setOfWords2Vec(myVocabList, listOPosts[3]))

#朴素贝叶斯分类器训练函数， 计算p(wi|c1),p(wi|c0)
def trainNB0(trainMatrix, trainCategory): #0代表正常，1代表侮辱性评论
    numTrainDocs = len(trainMatrix) #数据的个数
    numWords = len(trainMatrix[0])#特征的个数
    pAbusive = sum(trainCategory) / float(numTrainDocs)#侮辱性评论的比例
    p0Num = numpy.ones(numWords)#分子
    p1Num = numpy.ones(numWords)
    p0Denom = 2.0#分母
    p1Denom = 2.0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Temp = p1Num / p1Denom
    p1Vect = [math.log(a) for a in p1Temp]
    p0Temp = p0Num / p0Denom
    p0Vect = [math.log(a) for a in p0Temp]
    return p0Vect, p1Vect, pAbusive

trainMat = []
for postinDoc in listOPosts:
    trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
# print(trainMat)

p0V, p1V, pAb = trainNB0(trainMat, listClasses)
# print(pAb)
# print(p0V)#长度为32
# print(p1V)

#朴素贝叶斯分类的函数
def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = sum(vec2Classify * p1Vec) + math.log(pClass1)#元素相乘再加上侮辱性文档比例的对数
    p0 = sum(vec2Classify * p0Vec) + math.log(1 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0

def testingNB():
    listOPosts, listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    trainMat = []
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    p0V, p1V, pAb = trainNB0(numpy.array(trainMat), numpy.array(listClasses))
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = numpy.array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry, 'classified as:', classifyNB(thisDoc, p0V, p1V, pAb))
    testEntry = ['stupid', 'garbage']
    thisDoc = numpy.array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry, 'classified as:', classifyNB(thisDoc, p0V, p1V, pAb))

# testingNB()
#朴素贝叶斯词袋模型
def bagOfWords2VecMN(vocabList, inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec

# print(bagOfWords2VecMN(myVocabList, ['my', 'dog', 'my', 'dog']))

#切分文本
# mySent = 'This book is the best book on python or M.L. Ihave ever laid eyes upon.'
# print(mySent.split())
import re
regEx = re.compile('\\W*')
# listOfTokens = regEx.split(mySent)
# # print(listOfTokens)
# listOfTokens = [tok for tok in listOfTokens if len(tok) > 0]
# # print(listOfTokens)
# listOfTokens = [tok.lower() for tok in listOfTokens if len(tok) > 0]
# # print(listOfTokens)

emailText = open('F:/Python/机器学习实战/MLiA_SourceCode/machinelearninginaction/Ch04/email/ham/6.txt').read()
listOfTokens = regEx.split(emailText)
# print(listOfTokens)
# print(len(listOfTokens))

#测试算法：使用朴素贝叶斯进行交叉验证
#文本解析
def textParse(bigString):
    import re
    listOfTokens = re.split(r'\W*', bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]

def spamTest():
    docList = []
    classList = []
    fullText = []
    #读取50个邮件
    for i in range(1, 26):
        wordList = textParse(open(r'F:/Python/机器学习实战/MLiA_SourceCode/machinelearninginaction/Ch04/email/spam/%d.txt' %(i)).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(open(r'F:/Python/机器学习实战/MLiA_SourceCode/machinelearninginaction/Ch04/email/ham/%d.txt' %(i)).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)
    trainingSet = list(range(50))
    testSet = []
    #选取十个数据集作为测试集
    for i in range(20):
        randIndex = int(random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    trainMat = []
    trainClasses = []
    for docIndex in trainingSet:
        trainMat.append(setOfWords2Vec(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V, p1V, pSpam = trainNB0(numpy.array(trainMat), numpy.array(trainClasses))
    errorCount = 0
    for docIndex in testSet:
        wordVector = setOfWords2Vec(vocabList, docList[docIndex])
        if classifyNB(numpy.array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
            errorCount += 1
    print("the error rate is :", float(errorCount) / len(testSet))

# spamTest()

ny = feedparser.parse('http://newyork.craigslist.org/stp/index.rss')
print(ny['entries'])

#计算出前三十个最频繁的词
def calcMostFreq(vocabList, fullText):
    import operator
    freqDict = {}
    for token in vocabList:
        freqDict[token] = fullText.count(token)
    sortedFreq = sorted(freqDict.items(), key=operator.itemgetter(1), reverse=True)
    return sortedFreq[:30]

#高频去词函数
def localWords(feed1, feed0):
    import feedparser
    docList = []
    classList = []
    fullText = []
    minLen = min(len(feed1['entries']), len(feed0['entries']))
    for i in range(minLen):
        wordList = textParse(feed1['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(feed0['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)
    top30Words = calcMostFreq(vocabList, fullText)
    for pairw in top30Words:
        if pairw[0] in vocabList:
            vocabList.remove(pairw[0])
    trainingSet = range(2 * minLen)
    testSet = []
    for i in range(20):
        randIndex = int(random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    trainMat = []
    trainClasses = []
    for docIndex in trainingSet:
        trainMat.append(bagOfWords2VecMN(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V, p1V, pSpam = trainNB0(trainMat, trainClasses)
    errorCount = 0
    for docIndex in testSet:
        wordVector = bagOfWords2VecMN(vocabList, docList[docIndex])
        if classifyNB(numpy.array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
            errorCount += 1
    print("the error rate is :", float(errorCount) / len(testSet))
    return vocabList, p0V, p1V

ny = feedparser.parse('http://newyork.craigslist.org/stp/index.rss')
sf = feedparser.parse('http://sfbay.craigslist.org/stp/index.rss')
# vocabList, pSF, pNY = localWords(ny, sf)
# vocabList, pSF, pNY = localWords(ny, sf)

#最具表征性的词汇显示函数
def getTopWords(ny, sf):
    import feedparser
    vocabList, p0V, p1V = localWords(ny, sf)
    topNY = []
    topSF = []
    for i in range(len(p0V)):
        if p0V[i] > -6.0:
            topSF.append((vocabList[i], p0V[i]))
        if p1V[i] >= -6.0:
            topNY.append((vocabList[i], p1V[i]))
    sortedSF = sorted(topSF, key=lambda pair: pair[1], reverse=True)
    print("SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF")
    for item in sortedSF:
        print(item[0])
    sortedNY = sorted(topNY, key=lambda pair: pair[1], reverse=True)
    print("NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**")
    for item in sortedNY:
        print(item[0])

























