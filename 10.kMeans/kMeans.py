from sklearn.cluster import KMeans
from numpy import *
import matplotlib.pyplot as plt

# 将文本文件导入到一个列表中
# 该返回值是一个包含许多其他列表的列表
def loadDataSet(fileName):
    dataMat = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = []
        for i in range(len(curLine)):
            fltLine.append(float(curLine[i]))
        # fltLine = map(float,curLine)
        dataMat.append(fltLine)
    return dataMat

def main1():
    datMat = mat(loadDataSet('testSet.txt'))
    # 调用kmeans类
    numClust = 4
    clf = KMeans(n_clusters=numClust)
    s = clf.fit(datMat)
    print(s)
    # 质心
    print(clf.cluster_centers_)
    # 每个样本所属的簇
    print(clf.labels_)
    # 用来评估簇的个数是否合适，距离越小说明簇分的越好，选取临界点的簇个数
    print(clf.inertia_)
    # 进行预测
    print(clf.predict(datMat))
    scatterMarkers = ['s', 'o', '^', '8', 'p', 'd', 'v', 'h', '>', '<']
    for i in range(numClust):
        ptsInCurrCluster = datMat[nonzero(clf.labels_ == i)[0],:]
        markerStyle = scatterMarkers[i % len(scatterMarkers)]
        plt.scatter(ptsInCurrCluster[:, 0].tolist(), ptsInCurrCluster[:, 1].tolist(), marker=markerStyle,s=90)
    plt.scatter(clf.cluster_centers_[:, 0].tolist(), clf.cluster_centers_[:, 1].tolist(), marker='+', s=300)
    plt.show()

if __name__ == '__main__':
    main1()