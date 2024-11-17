import numpy as np
from TestTrees import *
NUM_CLASSES = 6
def sample(x,y,sampleproportion):
	sample_size = int(sampleproportion*len(x))
	rng = np.random.randint(len(x)-1, size=sample_size)
	x_samples = np.zeros((sample_size,x.shape[1]))
	y_samples = np.zeros((sample_size,1))
	for sample_idx in range(len(rng)):
		x_samples[sample_idx] = x[rng[sample_idx]]
		y_samples[sample_idx] = y[rng[sample_idx]]
	return x_samples,y_samples

def sample_attributes(attributes,size):
	rng = np.random.choice(len(attributes), size, replace=False)
	attributes_set = []
	for idx in rng:
		attributes_set.append(attributes[idx])
	return attributes_set

def decision_forest_vote(forest,x):
	decisions = np.zeros(x.shape[0])
	for datum_idx in range(len(x)):
		decision = 0
		for tree in forest:
			treeResult, depth = RunRowByTreeByDepth(tree, x[datum_idx])
			decision += treeResult
		decisions[datum_idx] = decision/float(len(forest))
	return decisions

def decision_forest_vote_adaboost(forest,class_n,x,y):
	weights = np.zeros(len(forest))
	for datum_idx in range(len(x)):
		votes = np.zeros(len(forest))
		for tree_idx in range(len(forest)):
			treeResult, depth = RunRowByTreeByDepth(forest[tree_idx], x[datum_idx])
			votes[tree_idx] = treeResult

		zeros = np.count_nonzero(votes==0)/float(len(forest))
		ones = np.count_nonzero(votes==1)/float(len(forest))
		for tree_idx in range(len(forest)):
			if class_n==y[datum_idx]:
				if votes[tree_idx] == 1:
					weights[tree_idx] += (ones - zeros)
				else:
					weights[tree_idx] -= NUM_CLASSES*(ones - zeros)					
			else:
				if votes[tree_idx] == 0:
					weights[tree_idx] += zeros - ones
				else:
					weights[tree_idx] -= NUM_CLASSES*(zeros - ones)
	weights = weights/sum(weights[:])

	return weights

def decision_forest_vote_weighted(forest,weights,x):
	decisions = np.zeros(x.shape[0])
	for datum_idx in range(len(x)):
		decision = 0		
		treeUsed = 0
		for tree_idx in range(len(forest)):
			if weights[tree_idx] > 0:
				treeResult, depth = RunRowByTreeByDepth(forest[tree_idx], x[datum_idx])
				treeUsed += 1
				if treeResult == 0:
					treeResult = -1
				decision += treeResult*weights[tree_idx]
		decisions[datum_idx] = decision/float(treeUsed)
	return decisions

def top_n_indexes(arr, n):
	
	arr = arr.reshape(arr.shape[0]*arr.shape[1])
	arr=arr.argsort()[-n:][::-1]
	return arr

def map_param(idx,p_size,a_size):
		return int(idx/a_size), idx%p_size
		