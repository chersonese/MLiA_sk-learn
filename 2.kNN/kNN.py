from numpy import *
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier

def file2matrix(filename):
    fr = open(filename)
    arrayOLines = fr.readlines()
    numberOfLines = len(arrayOLines)            #得到文件行数
    returnMat = zeros((numberOfLines,3))            #创建返回的NumPy矩阵
    classLabelVector = []
    index = 0
    for line in arrayOLines:
        line = line.strip()         #去除首尾空格
        listFromLine = line.split('\t')         #切片
        returnMat[index,:] = listFromLine[0:3]      #样本矩阵
        classLabelVector.append(int(listFromLine[-1]))      #样本标签
        index += 1
    return returnMat,classLabelVector

#归一化特征值
def autoNorm(dataSet):
    minVals = dataSet.min(0)        #参数0使得函数可以从列中选取最小值，而不是选取当前行的最小值。
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals,(m,1))
    normDataSet = normDataSet/tile(ranges,(m,1))
    return normDataSet,ranges,minVals       #返回归一化后的矩阵，范围，1*3最小值向量

#分类器针对约会网站的测试代码
def datingClassTest():
    hoRatio = 0.10      #设定用来测试的样本占比
    datingDataMat,datingLabels = file2matrix('datingTestSet2.txt')      #从文本中提取得到数据特征，及对应的标签
    normMat,ranges,minVals = autoNorm(datingDataMat)                #归一化数据
    m = normMat.shape[0]        #得到第一维度的大小
    numTestVecs = int(m*hoRatio)    #测试样本数量
    clf = KNeighborsClassifier(3,weights='uniform')
    clf.fit(normMat[numTestVecs:m,:],datingLabels[numTestVecs:m])
    Z = clf.predict(normMat[:numTestVecs])
    print("the total error rate is %f" % (1 - sum(Z == datingLabels[:numTestVecs])/ float(numTestVecs)))

'''
    errCount = 0.0      #错误数初始化
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],3)
        print("the classifier came back with:%d,the real answer is:%d"%(classifierResult,datingLabels[i]))
        if(classifierResult != datingLabels[i]):errCount += 1.0
    print("the total error rate is %f"%(errCount/float(numTestVecs)))
'''
#约会网站预测函数
def classifyPerson():
    resultList = ['not at all','in small doses','in large doses']       #定义预测结果
    percentTats = float(input("percentage of time spent playing video games?"))     #输入玩视频游戏所耗时间百分比
    ffMiles = float(input("frequent flier miles earned per year?"))                 #输入每年获得的飞行常客里程数
    iceCream = float(input("liters of ice cream consumed per year?"))               #输入每周消耗的冰淇淋公升数
    datingDataMat,datingLabels = file2matrix('datingTestSet2.txt')
    normMat,ranges,minVals = autoNorm(datingDataMat)
    inArr = array([ffMiles,percentTats,iceCream])       #将输入的数值放在数组中
    clf = KNeighborsClassifier(3,weights='uniform')
    clf.fit(normMat,datingLabels)
    Z = clf.predict(((inArr-minVals)/ranges).reshape(1,-1))
    print("You will probably like this person:", resultList[Z[0]-1])

'''
    classifierResult = classify0((inArr-minVals)/ranges,normMat,datingLabels,3)
    print("You will probably like this person:",resultList[classifierResult - 1])
'''

#手写数字识别系统的sklearn代码
digits = datasets.load_digits()     #读取数据集，数据集里的属性有data，target，target_name，images
totalNum = len(digits.data)
trainNum = int(0.9 * totalNum)
trainX = digits.data[0 : trainNum]       # 选出90%样本作为训练样本，其余10%测试
trainY = digits.target[0 : trainNum]
testX = digits.data[trainNum:]      #读取测试样本
testY = digits.target[trainNum:]
clf = KNeighborsClassifier(10, weights='uniform')
clf.fit(trainX, trainY)
Z = clf.predict(testX)          #Z是一个矩阵
print("\nthe total error rate is: %f" % ( 1 - sum(Z==testY) / float(len(testX))))