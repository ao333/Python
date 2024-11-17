import numpy as np
import RF as rt
import Bag as bl
import matplotlib.pyplot as plt


class BagLearner(object):

    def __init__(self, learner=rt.RTLearner, kwargs={}, bags=10, boost=False, verbose=False):
        self.learners = []
        self.bags = bags
        self.boost = boost
        self.verbose = verbose
        for i in range(self.bags):
            self.learners.append(learner(**kwargs))

    def addEvidence(self, dataX, dataY):
        for i in range(self.bags):
            self.learners[i].addEvidence(dataX, dataY)
            if self.verbose:
                print "Built Bag!"
                print self.learners

    def query(self, points):
        Y = []
        for i in range(self.bags):
            Y.append(self.learners[i].query(points))
            if self.verbose:
                print "found Y!"
                print Y[i]
        return np.mean(Y, axis=0)


if __name__ == '__main__':
    data = np.genfromtxt('data/Istanbul.csv', delimiter=',')
    data = data[1:, 1:]  # eliminate 1st row and column containing col-labels and dates
    np.random.shuffle(data)  # shuffle in-place
    split = int(0.6*data.shape[0])  # 60-40 break into train-test sets
    trainX = data[:split, :-1]
    trainY = data[:split, -1]  # last column is labels
    testX = data[split:, :-1]
    testY = data[split:, -1]  # last column is labels

    # create a BagLearner of RTlearners and train them
    learner = bl.BagLearner(learner=rt.RTLearner, kwargs={"leaf_size": 10}, \
                            bags=20, boost=False, verbose=False)  # create BagLearner
    learner.addEvidence(trainX, trainY)  # train BagLearner

    Y = learner.query(trainX)  # get the predictions
    rmse = np.sqrt(((Y - trainY)**2).sum() / trainY.shape[0])
    corr = np.corrcoef(Y, trainY)
    print("In sample results")
    print("RMSE: ", rmse)
    print("corr: ", corr[0, 1])

    Y = learner.query(testX)  # get the predictions
    rmse = np.sqrt(((Y - testY)**2).sum() / testY.shape[0])
    corr = np.corrcoef(Y, testY)
    print("Out of sample results")
    print("RMSE: ", rmse)
    print("corr: ", corr[0, 1])

    # Plot RMSE for training dataset vs. number of bags each containing an RTLearner
    maxBagSize = 20
    errTrain = np.zeros(maxBagSize)
    errTest = np.zeros(maxBagSize)
    reps = 10  # number of configurations to average over for RMSE calculation

    for size in range(maxBagSize):
        for i in range(reps):  # average by shuffling & selecting different train/test sets from mc3_p1.data
            np.random.shuffle(data)  # shuffle in-place
            learner = bl.BagLearner(learner=rt.RTLearner, kwargs={"leaf_size":10}, \
                                    bags=size + 1, boost = False, verbose=False)  # create BagLearner
            learner.addEvidence(trainX, trainY)  # train on shuffled trainX, trainY
            # training sample testing
            Y = learner.query(trainX)  # get the predictions
            errTrain[size] += np.sqrt(((Y - trainY)**2).sum() / trainY.shape[0])
            # test sample testing
            Y = learner.query(testX)  # get the predictions
            errTest[size] += np.sqrt(((Y - testY)**2).sum() / testY.shape[0])

    errTrain = errTrain / reps  # average over number of repetitions
    errTest = errTest / reps
    plt.plot(range(1, maxBagSize+1), errTrain, label='training')
    plt.plot(range(1, maxBagSize+1), errTest, label='test')
    plt.xlabel('number of bags')
    plt.ylabel('root mean squared error')
    plt.title('Bag Learner with leaf size = 10')
    lg = plt.legend(loc='best')
    lg.draw_frame(False)  # removes box around legend
    plt.show()
