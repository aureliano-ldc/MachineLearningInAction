'''
kNN: k Nearest Neighbors
========================
Pros: High accuracy, insensitive to outliers, no assumptions about data
Cons: Computationally expensive, requires a lot of memory
Works with: Numeric values, nominal values
========================

Input:      inX: vector to compare to existing dataset (1xN)
            dataSet: size m data set of known vectors (NxM)
            labels: data set labels (1xM vector)
            k: number of neighbors to use for comparison (should be an odd number)
            
Output:     the most popular class label

教程来自：<Machine Learning in Action>
            —— Classifying with K-Nearest Neoghbors
'''

from numpy import *
import operator
from os import listdir

""" K-近邻算法：
    输入向量和已知向量的距离为Euclidian距离，
    即2个向量间各维度差的平方和再开方；
    比如 (1,0,0,1) 和 (7, 6, 9, 4) 的距离为：
    [(7-1)^2 + (6-0)^2 + (9-0)^2 + (4-1)^2] ^0.5；
    求出输入向量和每一个已知向量的距离，取距离最近的K个，
    取K个中出现次数最多的类为输入向量的类。
"""
def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0] #取已知数据集第一维的大小，即行数
    # 求输入向量和各已知向量间的距离：
    diffMat = tile(inX, (dataSetSize,1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    # 按距离大小赋予序号：
    sortedDistIndicies = distances.argsort()
    # 统计这K个数据各类出现的次数
    classCount={}          
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

# argsort()函数将数组中的元素从小到大排列，提取其对应的index(索引)，然后输出到一个新的数组：
x = array([1,4,3,-1,6,9])
x.argsort()

# shape() 方法返回数组各维度的长度,
# 比如一个维度2、2、3的数组
x = array([[[1, 2, 3], [9, 8, 7]],
           [[4, 6, 7], [1, 1, 1]]])
x.shape[0]
x.shape[1]
x.shape[2]

# tile(A,reps)：A沿各个维度重复的次数, A和reps都是array_like型
A=[1,2] 
tile(A,2) 
tile(A,(2,2,3)) # 第1维3倍输出，第二维和第三维2倍输出
tile(A,(2,1)) 

# 用一个小例子测试一下：
def createDataSet():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group, labels

group, labels = createDataSet()
# 得到输入变量的类别为 'B'
classify0([0,0], group, labels, 3)


"""例一：用KNN改进约会网站的对象匹配：
    Hellen 收集了一些约会对象的数据，放在datingTestSet.txt，
    前3列是约会对象的特征：
    ■ Number of frequent flyer miles earned per year
    ■ Percentage of time spent playing video games
    ■ Liters of ice cream consumed per week
    最后一列是根据Hellen的中意程度对对象作的分类：
    ■ People she didn’t like
    ■ People she liked in small doses
    ■ People she liked in large doses
"""
# 先做一些数据处理的准备工作：
def file2matrix(filename):
    fr = open(filename)
    numberOfLines = len(fr.readlines())         #得到行数
    returnMat = zeros((numberOfLines,3))        #构建特征矩阵
    classLabelVector = []                       #构建类数组   
    fr = open(filename)
    index = 0
    for line in fr.readlines():
        line = line.strip()                     #去掉每行前后的空格
        listFromLine = line.split('\t')
        returnMat[index,:] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat,classLabelVector
    
datingDataMat,datingLabels = file2matrix('datingTestSet2.txt')

# 以第二列和第三列作散点图看看：
import matplotlib
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)
plt.xlabel("Percentage of time spent playing video games", fontsize=14)
plt.ylabel("Liters of ice cream consumed per week", fontsize=14)
# scatter()的后2个参数是size和color，
# 因为传入的是和散点总数一样大的数组，所以每个点的size和color是定制的
ax.scatter(datingDataMat[:,1], datingDataMat[:,2],
            15.0*array(datingLabels), 15.0*array(datingLabels))

plt.show() # 可以看到纵轴（第三个特征）似乎对分类没什么影响

# 以第一列和第二列作散点图看看：
fig = plt.figure()
ax = fig.add_subplot(111)
plt.xlabel("Number of frequent flyer miles earned per year", fontsize=14)
plt.ylabel("Percentage of time spent playing video games", fontsize=14)
ax.scatter(datingDataMat[:,0], datingDataMat[:,1],
            15.0*array(datingLabels), 15.0*array(datingLabels))
plt.show() 


""" 第一个特征其（飞行里程数）的数值比另两个大得多，
    这样在计算距离时第一个特征起决定性作用，其它两个变得可有可无，
    用原文的话： It shouldn’t have any extra importance, unless we want it to, 
    but Hellen believes these terms are equally important.
    所以我们要做 normalizing numeric values 的处理，
    对三个特征的数值区间作变换，压缩为 0-1 区间，
    公式： newValue = (oldValue-min)/(max-min)
"""
def autoNorm(dataSet):
    minVals = dataSet.min(0)  # 0是按列取极值，1是按行取极值
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m,1))
    normDataSet = normDataSet/tile(ranges, (m,1))   #element wise divide
    return normDataSet, ranges, minVals
   

""" 测试分类器的准确率
    我们用训练数据集的前50%作为测试数据
"""
def datingClassTest():
    hoRatio = 0.50      #hold out 10%
    datingDataMat,datingLabels = file2matrix('datingTestSet2.txt')       #load data setfrom file
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],3)
        print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, datingLabels[i]))
        if (classifierResult != datingLabels[i]): 
            errorCount += 1.0
    print("the total error rate is: %f" % (errorCount/float(numTestVecs)))
    print(errorCount)
    
# 整合成一个约会对象分类系统：
def classifyPerson():
    resultList = ['not at all', 'in small doses', 'in large doses']
    percentTats = float(input("percentage of time spent playing video games?"))
    ffMiles = float(input("frequent flier miles earned per year?"))
    iceCream = float(input("liters of ice cream consumed per year?"))
    datingDataMat,datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = array([ffMiles, percentTats, iceCream])
    classifierResult = classify0((inArr-minVals)/ranges,normMat,datingLabels,3)
    print("You will probably like this person: ", resultList[classifierResult - 1])






""" 例二：手写识别系统 """
# 把数字图像文档的32*32矩阵转换成分类器 classify0 可以接受的1*1024向量
def img2vector(filename):
    returnVect = zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32*i+j] = int(lineStr[j])
    return returnVect


testVector = img2vector('testDigits/0_13.txt')
testVector[0,0:31]
testVector[0,32:63]



# 定义手写识别系统
# 如果有看文档前面内容的话，这部分不难理解
def handwritingClassTest():
    hwLabels = []
    trainingFileList = listdir('trainingDigits')           #load the training set
    m = len(trainingFileList)
    trainingMat = zeros((m,1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]     #take off .txt
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i,:] = img2vector('trainingDigits/%s' % fileNameStr)
    testFileList = listdir('testDigits')        #iterate through the test set
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]     #take off .txt
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, classNumStr))
        if (classifierResult != classNumStr): errorCount += 1.0
    print("\nthe total number of errors is: %d" % errorCount)
    print("\nthe total error rate is: %f" % (errorCount/float(mTest)))