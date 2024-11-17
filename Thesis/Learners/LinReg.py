import numpy as np


class LinRegLearner(object):

    def __init__(self, verbose=False):
        pass

    def addEvidence(self, dataX, dataY):  # Add training mc3_p1.data to learner
        newdataX = np.ones([dataX.shape[0], dataX.shape[1]+1])
        # concatenate a 1s column so linear regression finds a constant term
        newdataX[:, 0:dataX.shape[1]] = dataX
        # build and save the model
        self.model_coefs, residuals, rank, s = np.linalg.lstsq(newdataX, dataY)
        
    def query(self, points):  # Estimate a set of test points given the model we built.
        # points: should be a numpy array with each row corresponding to a specific query.
        return (self.model_coefs[:-1] * points).sum(axis=1) + self.model_coefs[-1]
        # returns the estimated values according to the saved model.
