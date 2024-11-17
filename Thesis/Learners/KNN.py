import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class KNNLearner(object):
    
    def __init__(self, k=3):
        self.k = k

    def addEvidence(self, Xtrain, Ytrain):
        self.Xtrain = Xtrain
        self.Ytrain = Ytrain

    def query(self, Xtest):
        Y = np.zeros((Xtest.shape[0], 1), dtype='float')
        for i in range(Xtest.shape[0]):
            dist = (self.Xtrain[:, 0] - Xtest[i, 0])**2 + (self.Xtrain[:, 1] - Xtest[i, 1])**2
            knn = [self.Ytrain[knni] for knni in np.argsort(dist)[:self.k]]
            Y[i] = np.mean(knn)

        return Y


if __name__ == "__main__":
    # data = np.genfromtxt('data/best4lrr_data.csv', delimiter=',')
    dataTrain = np.genfromtxt('data/3_groups.csv', delimiter=',')
    dataTest = np.genfromtxt('data/ripple.csv', delimiter=',')

    # compute how much of the data is training and testing
    # train_rows = int(0.60* mc3_p1.data.shape[0]) #math.floor(0.60* mc3_p1.data.shape[0]) AJ commented out 'floor'
    # test_rows = mc3_p1.data.shape[0] - train_rows

    # separate out training and testing mc3_p1.data
    # trainX = mc3_p1.data[:train_rows,0:-1]
    # trainY = mc3_p1.data[:train_rows,-1]
    # testX = mc3_p1.data[train_rows:,0:-1]
    # testY = mc3_p1.data[train_rows:,-1]
    trainX = dataTrain[:, 0:-1]
    trainY = dataTrain[:, -1]
    testX = dataTest[:, 0:-1]
    testY = dataTest[:, -1]

    # create a learner and train it
    learner = KNNLearner(k=3)  # create a KNNLearner
    learner.addEvidence(trainX, trainY)  # train it

    # evaluate in sample
    Y = learner.query(trainX)  # get the predictions
    rmse = math.sqrt(((trainY - Y) ** 2).sum()/trainY.shape[0])
    print("In sample results")
    print("RMSE: ", rmse)
    # corr = np.corrcoef(Y, y=trainY)
    # print("corr: ", corr[0, 1])

    # evaluate out of sample
    Y = learner.query(testX)  # get the predictions
    rmse = math.sqrt(((testY - Y) ** 2).sum()/testY.shape[0])
    print
    print("Out of sample results")
    print("RMSE: ", rmse)
    # corr = np.corrcoef(Y, y=testY)
    # print("corr: ", corr[0, 1])
    print 'number of dependent variables = {}'.format(np.shape(trainX)[1])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(trainX[:, 0], trainX[:, 1], trainY, c='c')
    ax.scatter(testX[:, 0], testX[:, 1], testY, c='r')

    plt.subplot(121)
    plt.scatter(trainX[:, 0], trainY)
    plt.scatter(testX[:, 0], testY, c='r')
    plt.subplot(122)
    plt.scatter(trainX[:, 1], trainY)
    plt.scatter(testX[:, 1], testY, c='r')
    plt.show()
