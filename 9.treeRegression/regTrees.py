import numpy as np
from numpy import *
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt


def loadDataSet(filename):
    dataMat = []
    fr = open(filename)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        # 将每行映射成浮点数
        fltLine = []
        for i in range(len(curLine)):
            fltLine.append(float(curLine[i]))
        # fltLine = map(float,curLine)
        dataMat.append(fltLine)
    return dataMat

rawDat = mat(loadDataSet('sine.txt'))
testDat = arange(min(rawDat[:,0]),max(rawDat[:,1]),0.01)
x = rawDat[:,0].tolist()
y = rawDat[:,1].tolist()
X = testDat.reshape(-1,1).tolist()
clf = DecisionTreeRegressor(max_depth=5)
clf.fit(x,y)
Y = clf.predict(X)
plt.subplot(111)
plt.scatter(x,y,s=5)
plt.plot(testDat,Y,linewidth=2)
plt.show()