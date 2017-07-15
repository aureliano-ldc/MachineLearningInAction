"""
Logistic regression:
====================
Pros: Computationally inexpensive, easy to implement, knowledge representation
easy to interpret
Cons: Prone to underfitting, may have low accuracy
Works with: Numeric values, nominal values
====================

关于’回归‘的含义，原文这么说：
The regression aspects means that we try to find a best-fit set of parameters. 
Finding the best fit is similar to regression, and in this method it’s how we 
train our classifier. We’ll use optimization algorithms to find these best-fit parameters. 
This best-fit stuff is where the name regression comes from. 

梯度下降法找最小值，梯度上升法找最大值，
<统计学习方法>用梯度下降，本教程用梯度上升，殊途同归。

教程来自：<Machine Learning in Action>
            —— Logistic regression
"""
from numpy import *


def loadDataSet():
	dataMat = []; labelMat = []
	fr = open('Ch05/testSet.txt')
	for line in fr.readlines():
		lineArr = line.strip().split()
		dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
		labelMat.append(int(lineArr[2]))
	return dataMat, labelMat

# 在二类问题中，sigmoid function 决定输出的是0还是1
def sigmoid(inX):
	return 1.0/(1 + exp(-inX))

# 梯度上升法(Gradient ascent)：
def gradAscent(dataMatIn, classLabels):
	dataMatrix = mat(dataMatIn)                 # 转换成矩阵格式
	labelMat = mat(classLabels).transpose()     # 矩阵转置
	m, n = shape(dataMatrix)
	alpha = 0.001
	maxCycles = 500
	weights = ones((n, 1))
	for k in range(maxCycles):
		h = sigmoid(dataMatrix*weights)         # 矩阵乘法：（100·3）*（3·1）= 100·1
		error = (labelMat - h)
		weights = weights + alpha * dataMatrix.transpose() * error
		# 算法公式推导参考：http://blog.csdn.net/softimite_zifeng/article/details/53157218
	return weights

# 矩阵乘法举例：
A = mat(ones((3, 3)))
B = mat([1, 2, 3]).transpose()
A
B
A * B

# sigmoid()的输入可以是一个序列：
sigmoid(A*B)

# 执行
dataArr, labelMat = loadDataSet()
weights = gradAscent(dataArr,labelMat)



""" 作图：数据和它的最优拟合线 """
def plotBestFit(wei):
	import matplotlib.pyplot as plt
	if type(wei) == matrix:
		weights = wei.getA()
	else:
		weights = wei
	dataMat, labelMat = loadDataSet()
	dataArr = array(dataMat)
	n = shape(dataArr)[0]
	xcord1 = []; ycord1 = []
	xcord2 = []; ycord2 = []
	for i in range(n):
		if int(labelMat[i]) == 1:
			xcord1.append(dataArr[i, 1]); ycord1.append(dataArr[i, 2])
		else:
			xcord2.append(dataArr[i, 1]); ycord2.append(dataArr[i, 2])
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
	ax.scatter(xcord2, ycord2, s=30, c='green')
	x = arange(-3.0, 3.0, 0.1)
	y = (-weights[0] - weights[1]*x)/weights[2] 
	# 因为用的是梯度上升，不是梯度下降，所以参数前面加负号？
	ax.plot(x, y)
	plt.xlabel('X1'); plt.ylabel('X2');
	plt.show()


# help(matrix.getA): 
# 把矩阵作为数组返回，相当于np.asarray()
x = matrix(arange(12).reshape((3, 4)))
x.getA()

# 执行
plotBestFit(weights) # 拟合效果相当好，只有3、4个点错分类



""" 随机梯度上升法（stochastic gradient ascent）：
梯度上升（下降）每次更新会遍历整个数据集，
如果数据集有数以亿计的点数据和成千上百的特征，那么处理量太大，
所以我们用随机梯度法，每次只更新新的数据。 """
def stocGradAscent0(dataMatrix, classLabels):
	m, n = shape(dataMatrix)
	alpha = 0.01
	weights = ones(n)
	for i in range(m):
		h = sigmoid(sum(dataMatrix[i]*weights))
		error = classLabels[i] - h
		weights = weights + alpha * error * dataMatrix[i]
	return weights

# matrix的结构起码要是2维，array可以是1维：
a1 = array([1, 2, 3])
a2 = array([1, 1, 1]).transpose()
shape(a1)
shape(a2)
shape(a1) == shape(a2)

m1 = mat([1, 2, 3])
m2 = mat([1, 1, 1]).transpose()
shape(m1)
shape(m2)
shape(m1) == shape(m2)

# 数组乘法和矩阵乘法是不一样的，
# gradAscent()把输入的数组转换成矩阵，sigmoid()里就不用加sum()
# stocGradAscent()没有作转换，sigmoid()里就要加sum()
a1 * a2
# [out]: array([1, 2, 3])

m1 * m2
# [out]: matrix([[6]])


# 执行
dataArr, labelMat = loadDataSet()
weights = stocGradAscent0(array(dataArr), labelMat)
plotBestFit(weights) # 这次的拟合直线差很多


""" 改进随机梯度法:
随机梯度法要经过多次迭代w才会趋于一个极限区间，而且会始终有区间波动，
我们做2个改进：
	1、alpha随着迭代次数递减，逼近0.01
	2、每次迭代不是按顺序遍历整个数据集，而是随机抽取数据长度的次数，以消除周期性波动 """
def stocGradAscent1(dataMatrix, classLabels, numIter=150):
	m, n = shape(dataMatrix)
	weights = ones(n)
	staticIter = []
	for j in range(numIter):
		dataIndex = list(range(m))
		staticIndex = []
		for i in range(m):
			alpha = 4/(1.0+j+i) + 0.01
			randIndex = int(random.uniform(0, len(dataIndex)))
			staticIndex.append(randIndex)
			h = sigmoid(sum(dataMatrix[randIndex]*weights))
			error = classLabels[randIndex] - h
			weights = weights + alpha * error * dataMatrix[randIndex]
			del(dataIndex[randIndex])
		staticIter.append(staticIndex)
	return weights, staticIter

# numpy.random.uniform(low, high, size=1):
# 从一个均匀分布[low,high)中随机采样，返回指定size个样本，注意定义域是左闭右开
random.uniform(1, 10, 5)

# 执行
dataArr, labelMat = loadDataSet()
weights, staticIter = stocGradAscent1(array(dataArr),labelMat)
plotBestFit(weights) # 改进的梯度法拟合效果不错

""" 深入理解函数，会发现每次迭代并不是随机遍历每一个数据,序列在前的数据被抽到的次数更多, 
	为了证明这一点，在原函数的基础上增加了一个迭代对象的统计列表staticIter
	列表每一项记录了一次迭代被抽到的数据序号,还可以加总统计词频看看 """

staticIter[0]
staticIter[1]
staticIter[2]

from collections import Counter
Counter(staticIter[0])
Counter(staticIter[10])

# 作柱形分布图
fre = []
for i in range(0, len(staticIter)):
	for j in range(0, len(staticIter[i])):
		fre.append(staticIter[i][j])

fre = Counter(fre)

import pygal
hist = pygal.Bar()
hist.title = "Frenquencies of datas excuted."
hist.x_labels = fre.keys()
hist.x_title = "Indices"
hist.y_title = "Frenquency"
hist.add('hhh', fre.values())
hist.render_to_file('logisticReg0.svg')



""" 实例：estimating horse fatalities from colic
		（统计马匹患腹绞痛后的死亡率）
	"""

def classifyVector(inX, weights):
	prob = sigmoid(sum(inX*weights))
	if prob > 0.5:
		return 1.0
	else:
		return 0.0

def colicTest():
	frTrain = open('horseColicTraining.txt')
	frTest = open('horseColicTest.txt')
	trainingSet = []
	trainingLabels = []
	for line in frTrain.readlines():
		currLine = line.strip().split('\t')
		lineArr = []
		for i in range(21):
			lineArr.append(float(currLine[i]))
		trainingSet.append(lineArr)
		trainingLabels.append(float(currLine[21]))
	trainWeights, staticIter = stocGradAscent1(array(trainingSet), trainingLabels, 500)
	errorCount = 0
	numTestVec = 0.0
	for line in frTest.readlines():
		numTestVec += 1.0
		currLine = line.strip().split('\t')
		lineArr = []
		for i in range(21):
			lineArr.append(float(currLine[i]))
		if int(classifyVector(array(lineArr), trainWeights)) != int(currLine[21]):
			errorCount +=1
	errorRate = (float(errorCount)/numTestVec)
	print("the error rate of this test is: %f" % errorRate)
	return errorRate


def multiTest():
	numTests = 10
	errorSum = 0.0
	for k in range(numTests):
		errorSum += colicTest()
	print("after %d iterations the average error rate is: %f" % (numTests, errorSum/float(numTests)))
