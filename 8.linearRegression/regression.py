from scipy import *
import matplotlib.pyplot as plt
from sklearn import linear_model

def loadDataSet(filename):
    numFeat = len(open(filename).readline().split('\t')) - 1
    dataMat = []
    labelMat = []
    fr = open(filename)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat,labelMat

# 线性回归
def standRegres():
    # 计算线性回归的回归系数
    xArr,yArr = loadDataSet('ex0.txt')
    clf = linear_model.LinearRegression(fit_intercept=False)
    clf.fit(xArr,yArr)
    # print("sklearn 里面线性回归训练得到的回归系数：\n",clf.coef_)
    ws = mat(clf.coef_.reshape((2,1)))
    return ws

def main1():
    # 线性回归的绘图
    xArr,yArr = loadDataSet('ex0.txt')
    xMat = mat(xArr)
    yMat = mat(yArr)
    ws = standRegres()
    yHat = xMat*ws
    print("皮尔逊相关系数计算预测值和真实值之间的相关性：\n",corrcoef(yHat.T, yMat))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xMat[:,1].flatten().A[0], yMat.T[:,0].flatten().A[0])
    xCopy = xMat.copy()
    xCopy.sort(0)
    yHat = xCopy*ws
    ax.plot(xCopy[:,1],yHat)
    plt.show()

def ridgeTest(xArr,yArr):
    xMat = mat(xArr)
    yMat = mat(yArr).T
    yMean = mean(yMat,0)
    yMat = yMat - yMean
    xMeans = mean(xMat,0)
    xVar = var(xMat,0)
    xMat = (xMat - xMeans)/xVar
    numTestPts = 30
    wMat = zeros((numTestPts,shape(xMat)[1]))
    xArr = array(xMat)
    yArr = array(yMat)
    for i in range(numTestPts):
        reg = linear_model.Ridge(alpha=exp(i-10), fit_intercept=False)
        reg.fit(xArr, yArr)
        ws = mat(reg.coef_.reshape(8,1))
        wMat[i,:] = ws.T
    return wMat

def main2():
    # 岭回归的绘图
    abX, abY = loadDataSet('abalone.txt')
    ridgeWeights = ridgeTest(abX, abY)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(ridgeWeights)
    plt.show()

# lasso回归
def main3():
    abX, abY = loadDataSet('abalone.txt')
    reg = linear_model.Lasso(alpha=0.1,fit_intercept=False)
    reg.fit(abX,abY)
    print("sklearn 里面Lasso回归训练得到的回归系数：\n", reg.coef_)


if __name__ == '__main__':
    main3()

