from sklearn.svm import SVC,NuSVC
from numpy import *
import numpy as np

#将图像转换为测试向量，将32*32的矩阵转化成1*1024的向量
def img2vector(filename):
    returnVect = zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect

#基于SVM的手写数字识别
def loadImages(dirName):
    from os import listdir
    hwLabels = []
    trainingFileList = listdir(dirName)           #load the training set
    m = len(trainingFileList)
    trainingMat = zeros((m,1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]     #take off .txt
        classNumStr = int(fileStr.split('_')[0])
        if classNumStr == 9: hwLabels.append(-1)
        else: hwLabels.append(1)
        trainingMat[i,:] = img2vector('%s/%s' % (dirName, fileNameStr))
    return trainingMat, hwLabels

def main():
    trainX,trainY = loadImages('trainingDigits')
    clf = SVC()
    # clf = NuSVC()
    clf.fit(trainX, trainY)
    testX,testY = loadImages('testDigits')
    Z = clf.predict(testX)
    print("\nthe total error rate is: %f" % ( 1 - np.sum(Z==testY) / float(len(testX))))

if __name__ == '__main__':
    main()