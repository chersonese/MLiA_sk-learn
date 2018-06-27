from sklearn.svm import SVC
from numpy import *
import numpy as np
from sklearn.svm import NuSVC

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


# SVC:支持向量分类，基于libsvm实现的（libsvm详情参考 或者百科），数据拟合的时间复杂度是数据样本的二次方，这使得他很难扩展到10000个数据集。
# 当输入是多类别时（SVM最初是处理二分类问题的），通过一对一的方案解决，当然也有别的解决办法，比如说（以下为引用）：
#
# SVM解决多分类问题的方法 :
# SVM算法最初是为二值分类问题设计的，当处理多类问题时，就需要构造合适的多类分类器。目前，构造SVM多类分类器的方法主要有两类：
# 一类是直接法，直接在目标函数上进行修改，将多个分类面的参数求解合并到一个最优化问题中，通过求解该最优化问题“一次性”实现多类分类。这种方法看似简单，但其计算复杂度比较高，实现起来比较困难，只适合用于小型问题中；
# 另一类是间接法，主要是通过组合多个二分类器来实现多分类器的构造，常见的方法有one-against-one和one-against-all两种。
# a.一对多法（one-versus-rest,简称1-v-r SVMs）。训练时依次把某个类别的样本归为一类,其他剩余的样本归为另一类，这样k个类别的样本就构造出了k个SVM。分类时将未知样本分类为具有最大分类函数值的那类。
# b.一对一法（one-versus-one,简称1-v-1 SVMs）。其做法是在任意两类样本之间设计一个SVM，因此k个类别的样本就需要设计k(k-1)/2个SVM。当对一个未知样本进行分类时，最后得票最多的类别即为该未知样本的类别。Libsvm中的多类分类就是根据这个方法实现的。
# c.层次支持向量机（H-SVMs）。层次分类法首先将所有类别分成两个子类，再将子类进一步划分成两个次级子类，如此循环，直到得到一个单独的类别为止。
# 对c和d两种方法的详细说明可以参考论文《支持向量机在多类分类问题中的推广》（计算机工程与应用。2004）
# d.其他多类分类方法。除了以上几种方法外，还有有向无环图SVM（Directed Acyclic Graph SVMs，简称DAG-SVMs）和对类别进行二进制编码的纠错编码SVMs。
#
# SVC参数解释 :
# （1）C: 目标函数的惩罚系数C，用来平衡分类间隔margin和错分样本的，default C = 1.0；
# （2）kernel：参数选择有RBF, Linear, Poly, Sigmoid, 默认的是"RBF";
# （3）degree：if you choose 'Poly' in param 2, this is effective, degree决定了多项式的最高次幂；
# （4）gamma：核函数的系数('Poly', 'RBF' and 'Sigmoid'), 默认是gamma = 1 / n_features;
# （5）coef0：核函数中的独立项，'RBF' and 'Poly'有效；
# （6）probablity: 可能性估计是否使用(true or false)；
# （7）shrinking：是否进行启发式；
# （8）tol（default = 1e - 3）: svm结束标准的精度;
# （9）cache_size: 制定训练所需要的内存（以MB为单位）；
# （10）class_weight: 每个类所占据的权重，不同的类设置不同的惩罚参数C, 缺省的话自适应；
# （11）verbose: 跟多线程有关，不大明白啥意思具体；
# （12）max_iter: 最大迭代次数，default = 1， if max_iter = -1, no limited;
# （13）decision_function_shape ： ‘ovo’ 一对一, ‘ovr’ 多对多  or None 无, default=None
# （14）random_state ：用于概率估计的数据重排时的伪随机数生成器的种子。
#  ps：7,8,9一般不考虑。

trainX,trainY = loadImages('trainingDigits')
clf = SVC()
clf.fit(trainX, trainY)
testX,testY = loadImages('testDigits')
Z = clf.predict(testX)
print("\nthe total error rate is: %f" % ( 1 - np.sum(Z==testY) / float(len(testX))))

# NuSVC（Nu-Support Vector Classification.）：核支持向量分类，和SVC类似，也是基于libsvm实现的，但不同的是通过一个参数空值支持向量的个数
# NuSVC参数
# nu：训练误差的一个上界和支持向量的分数的下界。应在间隔（0，1 ]。
# 其余同SVC

clf = NuSVC()
clf.fit(trainX,trainY)
Z= clf.predict(testX)
print("\nthe total error rate is: %f" % ( 1 - np.sum(Z==testY) / float(len(testX))))
