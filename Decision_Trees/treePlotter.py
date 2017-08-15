
""" 用matplotlib annotations 画出决策树 
	"""
import matplotlib.pyplot as plt 

# 定义方框和箭头的格式
# boxstyle定义框的边缘，sawtooth是波浪线型，fc控制的注解框内的颜色深度  
decisionNode = dict(boxstyle='sawtooth', fc='0.8') 
leafNode = dict(boxstyle='round4', fc='0.8')
arrow_args = dict(arrowstyle="<-")

def plotNode(nodeTxt, centerPt, parentPt, nodeType):
	# centerPt是方框坐标，parentPt是箭尾坐标，nodeTxt在框内
	createPlot.ax1.annotate(nodeTxt, xy=parentPt, 
							xycoords='axes fraction',
							xytext=centerPt, textcoords='axes fraction',
							va='center', ha='center',
							bbox=nodeType, arrowprops=arrow_args)


def createPlot():
	fig = plt.figure(1, facecolor='white')
	fig.clf() # clf(): clear the current figure
	createPlot.ax1 = plt.subplot(111, frameon=False)
	plotNode('a decision node', (0.5, 0.1), (0.1, 0.5), decisionNode)
	plotNode('a leaf node', (0.8, 0.1), (0.3, 0.8), leafNode)
	plt.show()

# 执行
createPlot()


""" 画之前，我们需要知道一棵树有多大，合理分配坐标空间
	测量叶节点的个数以分配X轴的区间
	测量决策树有几层以分配Y轴的区间
"""
# 返回决策树的叶节点树
def getNumLeafs(myTree):
	numLeafs = 0
	firstStr = list(myTree.keys())[0]
	secondDict = myTree[firstStr]
	for key in secondDict.keys():
		if type(secondDict[key]).__name__ == 'dict':
			numLeafs += getNumLeafs(secondDict[key])
		else:
			numLeafs += 1
	return numLeafs

# 返回决策树的深度（层数）
def getTreeDepth(myTree):
	maxDepth = 0
	firstStr = list(myTree.keys())[0]
	secondDict = myTree[firstStr]
	for key in secondDict.keys():
		if type(secondDict[key]).__name__ == 'dict':
			thisDepth = 1 + getTreeDepth(secondDict[key])
		else:
			thisDepth = 1
		if thisDepth > maxDepth:
			maxDepth = thisDepth
	return maxDepth

# 弄一颗测试树
def retrieveTree(i):
	listOfTrees = [{'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}},
				  {'no surfacing': {0: 'no', 1: {'flippers': {0: {'head': {0: 'no', 1: 'yes'}}, 1: 'no'}}}}]
	return listOfTrees[i]

# 执行
myTree = retrieveTree(0)
getNumLeafs(myTree)
getTreeDepth(myTree)


""" 真正开始画 """
# 在子节点和父节点之间插入文本
def plotMidText(cntrPt, parentPt, txtString):
	xMid = (parentPt[0] - cntrPt[0])/2.0 + cntrPt[0]
	yMid = (parentPt[1] - cntrPt[1])/2.0 + cntrPt[1]
	createPlot.ax1.text(xMid, yMid, txtString)

# 核心函数，X轴和Y轴区间是0-1
def plotTree(myTree, parentPt, nodeTxt):
	numLeafs = getNumLeafs(myTree)
	getTreeDepth(myTree)
	firstStr = list(myTree.keys())[0]
	# plotTree.totalW和plotTree.totalD是两个全局变量，用来保存树的深度和叶数
	# 作用是调整图在X轴和Y轴的位置，比如根节点的水平位置应在所有叶节点的中间
	cntrPt = (plotTree.xOff + (1.0 + float(numLeafs))/2.0/plotTree.totalW, plotTree.yOff)
	plotMidText(cntrPt, parentPt, nodeTxt)
	plotNode(firstStr, cntrPt, parentPt, decisionNode)
	secondDict = myTree[firstStr]
	# plotTree.xOff和plotTree.yOff也是两个全局变量，用来记录每次递归作图的位置
	# 当前节点的xOff取决与它旗下叶节点数占总数的比例，yOff简单点，根据深度递减就好
	plotTree.yOff = plotTree.yOff - 1.0/plotTree.totalD
	for key in secondDict.keys():
		if type(secondDict[key]).__name__ == 'dict':
			plotTree(secondDict[key], cntrPt, str(key))
		else:
			plotTree.xOff = plotTree.xOff + 1.0/plotTree.totalW
			plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
			plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
	plotTree.yOff = plotTree.yOff + 1.0/plotTree.totalD

""" 关于xOff的处理公式是该函数的难点:
	plotTree.xOff即为最近绘制的一个叶子节点的x坐标，在确定当前节点位置时每次只需确定当前节点有几个叶子节点，
	因此其叶子节点所占的总距离就确定了即为float(numLeafs)/plotTree.totalW*1(因为总长度为1)，
	因此当前节点的位置即为其所有叶子节点所占距离的中间即一半为float(numLeafs)/2.0/plotTree.totalW*1，
	但是由于开始plotTree.xOff赋值并非从0开始，而是左移了半个表格，因此还需加上半个表格距离
	即为1/2/plotTree.totalW*1,则加起来便为(1.0 + float(numLeafs))/2.0/plotTree.totalW*1，
	因此偏移量确定，则x位置变为plotTree.xOff + (1.0 + float(numLeafs))/2.0/plotTree.totalW
"""

def createPlot(inTree):
	fig = plt.figure(1, facecolor='white')
	fig.clf()
	axprops = dict(xticks=[], yticks=[])
	createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)
	plotTree.totalW = float(getNumLeafs(inTree))
	plotTree.totalD = float(getTreeDepth(inTree))
	plotTree.xOff = -0.5/plotTree.totalW
	plotTree.yOff = 1.0
	plotTree(inTree, (0.5, 1.0), '')
	plt.show()

# 执行
myTree = retrieveTree (0)
createPlot(myTree)

# 修改一下myTree再执行看看
myTree['no surfacing'][3] = 'maybe'
createPlot(myTree)



""" 定义决策树分类器 """
def classify(inputTree, featLabels, testVec):
	firstStr = list(inputTree.keys())[0]
	secondDict = inputTree[firstStr]
	featIndex = featLabels.index(firstStr)
	for key in secondDict.keys():
		if type(secondDict[key]).__name__ == 'dict':
			classLabel = classify(secondDict[key], featLabels, testVec)
		else:
			classLabel = secondDict[key]
	return classLabel

# 执行
# createDataSet() 生成的labels不是类名，而是特征名
import trees
myDat, labels = trees.createDataSet()
myTree = retrieveTree(0)
classify(myTree, labels, [1,0])




""" 
	实例：预测隐形眼镜(contact lens)分类：
	结果出现过拟合，需要剪枝，第9章再讲
"""
fr = open('lenses.txt', 'r')
lenses = [inst.strip().split('\t') for inst in fr.readlines()]
lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']

lensesTree = trees.createTree(lenses, lensesLabels)
createPlot(lensesTree)
