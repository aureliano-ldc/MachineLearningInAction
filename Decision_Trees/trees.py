"""
Decision Trees:
===============
Pros: Computationally cheap to use, easy for humans to understand learned results,
missing values OK, can deal with irrelevant features
Cons: Prone to overfitting
Works with: Numeric values, nominal values
===============

熵的概念用来衡量数据集中包含的信息量
(The measure of information of a set is known as entropy)

信息增益是数据集根据某个特征划分前和划分后熵的变化
(The change in information before and after the split is known as the information gain)

我们选择信息增益最大的特征作为划分的特征
(The split with the highest information gain is your best option)


教程来自：<Machine Learning in Action>
            —— Splitting datasets one feature at a time: decision trees

"""


# 定义一个计算熵的函数：
# 熵的详细介绍： http://m.blog.csdn.net/hguisu/article/details/27305435
from math import log

def calcShannonEnt(dataSet):
	numEntries = len(dataSet)
	labelCounts = {}
	for featVec in dataSet:
		currentLabel = featVec[-1]
		if currentLabel not in labelCounts.keys():
			labelCounts[currentLabel] = 1
		else:
			labelCounts[currentLabel] += 1
	shannonEnt = 0.0
	proSum = 0.0
	for key in labelCounts:
		prob = float(labelCounts[key])/numEntries
		proSum += prob
		shannonEnt -= prob * log(prob, 2)
	return shannonEnt  
	# proSum是分布的概率和，应该=1，用来验证计算是否正确,需要可以返回


""" 创建一个判断鱼类的简单数据，用来试验上面的熵函数，如下表所示：
	Can survive without
	coming to surface?          Has flippers?           Fish?
1 	Yes 						Yes 					Yes
2 	Yes 						Yes 					Yes
3 	Yes 						No 						No
4 	No 							Yes 					No
5 	No 							Yes 					No

"""
def createDataSet():
	dataSet = [[1, 1, 'yes'],
			   [1, 1, 'yes'],
			   [1, 0, 'no'],
			   [0, 1, 'no'],
			   [0, 1, 'no']]
	labels = ['no surfacing','flippers']
	return dataSet, labels

myDat,labels = createDataSet()

calcShannonEnt(myDat)


""" 根据某个特征作数据划分的函数：
输入：
		1.要划分的数据
		2.特征序号
		3.指定返回的特征值
		"""
def splitDataSet(dataSet, axis, value):
	retDataSet = []
	for featVec in dataSet:
		if featVec[axis] == value:
			# 子集里面不包含已经用来划分的特征
			reducedFeatVec = featVec[:axis]
			reducedFeatVec.extend(featVec[axis+1:])
			retDataSet.append(reducedFeatVec)
	return retDataSet

# 执行
myDat, labels = createDataSet()
myDat
splitDataSet(myDat, 0, 1)  #把数据中第一个特征值=1的选出来
splitDataSet(myDat, 0, 0)  #把数据中第一个特征值=0的选出来

# extend() 和 append() 的区别
a = [1,2,3]
b = [4,5,6]
a.append(b)
a
a = [1,2,3]
a.extend(b)
a


""" 有了计算熵的函数和划分集合的函数后，
	现在可以遍历所有特征，选择最优特征作划分 """
def chooseBestFeatureToSplit(dataSet):
	numFeatures = len(dataSet[0]) - 1
	baseEntropy = calcShannonEnt(dataSet) # 计算H(D)
	bestInfoGain = 0.0
	bestFeature = -1
	for i in range(numFeatures):
		featList = [example[i] for example in dataSet]
		uniqueVals = set(featList)
		newEntropy = 0.0
		# 下面计算条件熵
		for value in uniqueVals:
			subDataSet = splitDataSet(dataSet, i, value)
			prob = len(subDataSet)/float(len(dataSet))
			newEntropy += prob * calcShannonEnt(subDataSet)
		infoGain = baseEntropy - newEntropy
		if (infoGain > bestInfoGain):
			bestInfoGain = infoGain
			bestFeature = i 
	return bestFeature

# 执行
myDat, labels = createDataSet()
chooseBestFeatureToSplit(myDat) # 结果显示第一个特征最佳


""" 构建决策树:
	决策树不是二分树，
	选定的特征有几个不同的值就划分成几个子集
	不断对子集作同样的操作——选择最有特征，划分成更小的子集
	直到没有特征或该子集只包含一个类
	C4.5 和 CART 每次划分不会消耗特征，所以不会出现特征用完的情况，后面会讲到 """

# 如果出现特征用完的情况，按多数原则确定叶节点的类
def majorityCnt(classList):
	classCount = {}
	for vote in classList:
		if vote not in classCount.keys():
			classCount[vote] = 1
		else:
			classCount[vote] += 1
	sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
	return sortedClassCount[0][0]


def createTree(dataSet, labels):
	classList = [example[-1] for example in dataSet]
	# 如果只有一个类，那么返回类的值，结束
	if classList.count(classList[0]) == len(classList):
		return classList[0]
	# 如果没有特征了，那么按多数原则确定类, 结束
	if len(dataSet[0]) == 1:
		return majorityCnt(classList)
	# 否则，选择特征，建树
	bestFeat = chooseBestFeatureToSplit(dataSet)
	bestFeatLabel = labels[bestFeat]
	myTree = {bestFeatLabel:{}}
	del(labels[bestFeat])
	featValues = [example[bestFeat] for example in dataSet]
	uniqueVals = set(featValues)
	for value in uniqueVals:
		subLabels = labels[:]
		myTree[bestFeatLabel][value] = createTree(
										splitDataSet(dataSet, bestFeat, value), subLabels)
	return myTree

# 执行
myDat, labels = createDataSet()
myTree = createTree(myDat,labels)



