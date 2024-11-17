import numpy as np
import matplotlib.pyplot as plt


class RTLearner(object):

	def __init__(self, leaf_size=1, verbose=False):
		self.leaf_size = leaf_size
		self.verbose = verbose
		self.model = []

	def addEvidence(self, dataX, dataY):
		dataY = dataY[:, None]
		data = np.concatenate((dataX, dataY), axis=1)
		self.model = self.build_tree(data)
		if self.verbose:
			print "Built random tree!"
			print self.model[:, 0]

	def build_tree(self, data):
		if data.shape[0] <= self.leaf_size:
			return np.array([["Leaf", np.mean(data[:, -1]), "NA", "NA"]], dtype=object)
		if np.unique(data[:, -1]).shape[0] == 1:
			return np.array([["Leaf", np.unique(data[:, -1])[0], "NA", "NA"]], dtype=object)
		else:
			i = self.find_feature(data[:, 0:-1])
			r1 = np.random.randint(data.shape[0])
			r2 = np.random.randint(data.shape[0])
			SplitVal = (data[r1, i]+data[r2, i])/2
			d1 = data[data[:, i] <= SplitVal]
			if np.array_equal(d1,data):
				return np.array([["Leaf", np.mean(data[:,-1]), "NA", "NA"]], dtype=object)
			lefttree = self.build_tree(d1)
			righttree = self.build_tree(data[data[:, i] > SplitVal])
			root = np.array([[i, SplitVal, 1, lefttree.shape[0] + 1]], dtype=object)
			return np.concatenate((root, lefttree, righttree), axis=0)

	def find_feature(self, features):
		num = features.shape[1]
		index = np.random.randint(num)
		return index

	def query(self, points):
		num = points.shape[0]
		self.Y = []
		for i in range(num):
			self.query_helper(points[i], 0)
		if self.verbose:
			print "find Y successfully!"
			print np.array(self.Y)
		return np.array(self.Y)

	def query_helper(self, point, index):
		node = self.model[index, :]
		if node[0] == 'Leaf':
			self.Y.append(node[1])
			pass
		elif point[node[0]] <= node[1]:
			self.query_helper(point, index + node[2])
		elif point[node[0]] > node[1]:
			self.query_helper(point, index + node[3])
		elif self.verbose:
			print "Can't find leaf!"


if __name__ == '__main__':
	data = np.genfromtxt('data/Istanbul.csv', delimiter=',')
	data = data[1:, 1:]  # eliminate 1st row and 1st column containing col-labels and dates
	np.random.shuffle(data)  # shuffle in-place
	split = int(0.6*data.shape[0])  # 60-40 break into train-test sets
	trainX = data[:split, :-1]
	trainY = data[:split, -1]  # last column is labels
	testX = data[split:, :-1]
	testY = data[split:, -1]  # last column is labels

	# create a learner and train it
	learner = RTLearner(leaf_size=1, verbose=True)  # create an RTLearner
	learner.addEvidence(trainX, trainY)  # train it

	# in sample testing
	Y = learner.query(trainX) # get the predictions
	rmse = (np.sqrt(((Y - trainY)**2)/trainY.shape[0])).sum()
	corr = np.corrcoef(Y, trainY)
	print("In sample results")
	print("RMSE: ", rmse)
	print("corr: ", corr[0, 1])

	# out of sample testing
	Y = learner.query(testX)  # get the predictions
	rmse = (np.sqrt(((Y - testY)**2)/testY.shape[0])).sum()
	corr = np.corrcoef(Y, testY)
	print
	print("Out of sample results")
	print("RMSE: ", rmse)
	print("corr: ", corr[0, 1])

	# Plot RMSE for training dataset vs. leaf_size
	maxLeafSize = 10
	errTrain = np.zeros(maxLeafSize)
	errTest = np.zeros(maxLeafSize)
	for size in range(1, maxLeafSize+1):
		learner = RTLearner(leaf_size = size, verbose = False)  # create learner
		for i in range(100):
			np.random.shuffle(data)  # shuffle in-place
			learner.addEvidence(trainX, trainY)  # train on shuffled trainX, trainY
			# training sample testing
			Y = learner.query(trainX)  # get the predictions
			errTrain[size-1] += np.sqrt(((Y - trainY)**2).sum() / trainY.shape[0])
			# test sample testing
			Y = learner.query(testX)  # get the predictions
			errTest[size-1] += np.sqrt(((Y - testY)**2).sum() / testY.shape[0])

	plt.plot(range(1, maxLeafSize+1), errTrain/100, label='training')
	plt.plot(range(1, maxLeafSize+1), errTest/100, label='test')
	plt.xlabel('leaf size')
	plt.ylabel('root mean squared error')
	plt.title('Random Decision Tree')
	lg = plt.legend(loc='best')
	lg.draw_frame(False)
	plt.show()
