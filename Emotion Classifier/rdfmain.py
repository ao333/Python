import scipy.io
import sys
import numpy as np
from SplitSet import *
from PreProcess import *
from DecisionTree import *
from rdffunctions import *
from Visualizer import *
from TestTrees import *
from ConfusionMatrix import *
from Measure import *
from CountOccurrence import *
from DeepCopyTree import *
from PrintMatrix import *
from DataDump import *

import os.path

k_folds = False
validation = False
adaboost = True
randomsample = False

FULLTRAINING_ADABOOST = 1
FULLTRAINING_NOADABOOST = 2
KFOLDS_ADABOOST = 3
KFOLDS_NOADABOOST = 4
VALIDATIONMODE = 5




NUM_CLASSES = 6
NUM_FOLDS = 10
NUM_TREES_IN_FOREST = 300
ATTRIBUTE_SAMPLE_SIZE = 30 #13
SAMPLE_PROPORTION = 1.0
RANDOMSAMPLELOWER = 7
RANDOMSAMPLEUPPER = 32

VALIDATION_ATTRIBUTE_SIZE_LOWER = 7
VALIDATION_ATTRIBUTE_SIZE_UPPER = 32
VALIDATION_ATTRIBUTE_SIZE_INTERVAL = 1
VALIDATION_ATTRIBUTE_SIZE = (VALIDATION_ATTRIBUTE_SIZE_UPPER - VALIDATION_ATTRIBUTE_SIZE_LOWER)/VALIDATION_ATTRIBUTE_SIZE_INTERVAL + 1

SAMPLE_PROPORTION_LOWER = 1.0
SAMPLE_PROPORTION_UPPER = 1.0
SAMPLE_PROPORTION_INTERVAL = 0.05
SAMPLE_PROPORTION_SIZE = (SAMPLE_PROPORTION_UPPER-SAMPLE_PROPORTION_LOWER)/SAMPLE_PROPORTION_INTERVAL+1

PRINT_INTERVAL = 50


try:
    dataFilename = sys.argv[1]
    print 'loaded',dataFilename
except IndexError:
    dataFilename = 'cleandata_students.mat'
    print 'loaded default data', dataFilename
try:
	mode = int(sys.argv[2])
	if mode == FULLTRAINING_NOADABOOST:
		print 'Set mode: Full training with no AdaBoost'
		adaboost = False
	if mode == KFOLDS_ADABOOST:
		print 'Set mode: KFolds with AdaBoost'
		k_folds = True
	if mode == KFOLDS_NOADABOOST:
		print 'Set mode: KFolds with no AdaBoost'
		k_folds = True
		adaboost = False
	if mode == VALIDATIONMODE:
		print 'Set mode: Validation'
		print 'Attribute Lower Limit: ',VALIDATION_ATTRIBUTE_SIZE_LOWER,'Attribute Upper Limit:',VALIDATION_ATTRIBUTE_SIZE_UPPER,'Attribute Interval', VALIDATION_ATTRIBUTE_SIZE_INTERVAL
		validation = True
		k_folds = True

except IndexError:
	print 'Defualt mode: Full training with AdaBoost'


x,y = parse_data(scipy.io.loadmat(dataFilename))


hyper_parameter_cr = np.zeros((int(VALIDATION_ATTRIBUTE_SIZE),int(SAMPLE_PROPORTION_SIZE)))

stacked_sample_error_depth = 0
stacked_confusion_matrices_depth = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int32)
confusion_matrices_depth = []
if k_folds == True:
	for fold_idx in range(NUM_FOLDS):
		fold_num = fold_idx+1
		print 'Fold',fold_num
		(y_train, y_test) = SplitSet(y, fold_idx)
		(x_train, x_test) = SplitSet(x, fold_idx)

		if validation == True:
			y_predictions = np.zeros((y_test.shape[0], 1), dtype=np.int32)
			for sample_proportion_idx in range(int(SAMPLE_PROPORTION_SIZE)):
				sample_proportion = SAMPLE_PROPORTION_LOWER+sample_proportion_idx*SAMPLE_PROPORTION_INTERVAL
				for attribute_size_idx in range(VALIDATION_ATTRIBUTE_SIZE):
					attribute_size = VALIDATION_ATTRIBUTE_SIZE_LOWER+attribute_size_idx*VALIDATION_ATTRIBUTE_SIZE_INTERVAL
					forestList = []
					for class_idx in range(NUM_CLASSES):
						class_num = class_idx + 1
						forestList.append([])
						for tree_idx in range(NUM_TREES_IN_FOREST):
							x_sample,y_sample = sample(x_train,y_train,sample_proportion)
							train_targets = PreProcess(y_sample,class_num)
							attribute_set = sample_attributes(range(x.shape[1]),attribute_size)

							tree = Decision_Tree_Learning(x_sample,attribute_set,train_targets)

							forestList[class_idx].append(tree)

					vote_block = np.zeros((x_test.shape[0],NUM_CLASSES))
					for forest_idx in range(NUM_CLASSES):
						choices = decision_forest_vote(forestList[forest_idx],x_test)
						vote_block[:,forest_idx] = choices

					y_test_predictions = np.zeros((x_test.shape[0],1))

					for vote_idx in range(vote_block.shape[0]):
						y_test_predictions[vote_idx] = np.argmax(vote_block[vote_idx])+1

					depth_correct = 0
					for prediction_idx in range(0, y_test_predictions.shape[0]):
						if y_test_predictions[prediction_idx] == y_test[prediction_idx]:
							depth_correct += 1
					depth_correct /= float(len(y_test_predictions))

					hyper_parameter_cr[attribute_size_idx][sample_proportion_idx]+=depth_correct
					print 'sample portion ', sample_proportion, ' attribute size ', attribute_size, ' average CR ', hyper_parameter_cr[attribute_size_idx][sample_proportion_idx]/fold_num

		elif adaboost == True:
			(y_train, y_validate) = SplitSet(y_train, fold_idx)
			(x_train, x_validate) = SplitSet(x_train, fold_idx)

			forestList = []
			y_predictions = np.zeros((y_validate.shape[0], 1), dtype=np.int32)

			for class_idx in range(NUM_CLASSES):
				class_num = class_idx + 1
				forestList.append([])
				for tree_idx in range(NUM_TREES_IN_FOREST):
					x_sample,y_sample = sample(x_train,y_train,SAMPLE_PROPORTION)
					train_targets = PreProcess(y_sample,class_num)
					if randomsample == True:
						attribute_sample_size = np.random.randint(RANDOMSAMPLELOWER,RANDOMSAMPLEUPPER)
					else:
						attribute_sample_size = ATTRIBUTE_SAMPLE_SIZE
					attribute_set = sample_attributes(range(x.shape[1]),attribute_sample_size)
					tree = Decision_Tree_Learning(x_sample,attribute_set,train_targets)

					forestList[class_idx].append(tree)
					if (tree_idx+1)%PRINT_INTERVAL == 0:
						print 'Training Tree ', tree_idx+1, ' of forest ', class_num, ' of fold ', fold_num, ' attribute size ', attribute_sample_size

			forest_weights = np.zeros((NUM_TREES_IN_FOREST,NUM_CLASSES))
			for forest_idx in range(NUM_CLASSES):
				forest_weights[:,forest_idx] = decision_forest_vote_adaboost(forestList[forest_idx],forest_idx+1,x_validate,y_validate)

			vote_block = np.zeros((x_test.shape[0],NUM_CLASSES))
			for forest_idx in range(NUM_CLASSES):
				choices = decision_forest_vote_weighted(forestList[forest_idx],forest_weights[:,forest_idx],x_test)
				vote_block[:,forest_idx] = choices

			y_test_predictions = np.zeros((x_test.shape[0],1))

			for vote_idx in range(vote_block.shape[0]):
				y_test_predictions[vote_idx] = int(np.argmax(vote_block[vote_idx]))+1

			depth_correct = 0
			for prediction_idx in range(0, y_test_predictions.shape[0]):
				if y_test_predictions[prediction_idx] == y_test[prediction_idx]:
					depth_correct += 1

			depth_correct /= float(len(y_test_predictions))

			stacked_sample_error_depth += 1-depth_correct
	  		stacked_confusion_matrices_depth += ConfusionMatrix(y_test_predictions,y_test,NUM_CLASSES)
	  		confusion_matrices_depth.append(ConfusionMatrix(y_test_predictions,y_test,NUM_CLASSES))

	  		print 'Fold accuracy:',depth_correct




		else:
			forestList = []
			y_predictions = np.zeros((y_test.shape[0], 1), dtype=np.int32)
			for class_idx in range(NUM_CLASSES):
				class_num = class_idx + 1
				forestList.append([])
				for tree_idx in range(NUM_TREES_IN_FOREST):
					x_sample,y_sample = sample(x_train,y_train,SAMPLE_PROPORTION)
					train_targets = PreProcess(y_sample,class_num)
					if randomsample == True:
						attribute_sample_size = np.random.randint(RANDOMSAMPLELOWER,RANDOMSAMPLEUPPER)
					else:
						attribute_sample_size = ATTRIBUTE_SAMPLE_SIZE
					attribute_set = sample_attributes(range(x.shape[1]),attribute_sample_size)

					tree = Decision_Tree_Learning(x_sample,attribute_set,train_targets)

					forestList[class_idx].append(tree)
					if (tree_idx+1)%PRINT_INTERVAL == 0:
						print 'Training Tree ', tree_idx+1, ' of forest ', class_num, ' of fold ', fold_num, ' attribute size ', attribute_sample_size

			vote_block = np.zeros((x_test.shape[0],NUM_CLASSES))
			for forest_idx in range(NUM_CLASSES):
				choices = decision_forest_vote(forestList[forest_idx],x_test)
				vote_block[:,forest_idx] = choices

			y_test_predictions = np.zeros((x_test.shape[0],1))

			for vote_idx in range(vote_block.shape[0]):
				y_test_predictions[vote_idx] = int(np.argmax(vote_block[vote_idx]))+1

			depth_correct = 0
			for prediction_idx in range(0, y_test_predictions.shape[0]):
				if y_test_predictions[prediction_idx] == y_test[prediction_idx]:
					depth_correct += 1

			depth_correct /= float(len(y_test_predictions))
			print 'Fold accuracy:',depth_correct
			stacked_sample_error_depth += 1-depth_correct
	  		stacked_confusion_matrices_depth += ConfusionMatrix(y_test_predictions,y_test,NUM_CLASSES)
	    	confusion_matrices_depth.append(ConfusionMatrix(y_test_predictions,y_test,NUM_CLASSES))


if validation == True:
	indices = top_n_indexes(hyper_parameter_cr,5)

	print 'top 5 hyper-params'
	for index in indices:
		proportion_idx,attribute_idx = map_param(index,SAMPLE_PROPORTION_SIZE,VALIDATION_ATTRIBUTE_SIZE)
		print 'proportion ', SAMPLE_PROPORTION_LOWER+proportion_idx*SAMPLE_PROPORTION_INTERVAL, ' attribute ', VALIDATION_ATTRIBUTE_SIZE_LOWER+attribute_idx*VALIDATION_ATTRIBUTE_SIZE_INTERVAL
elif k_folds == False:
	if adaboost == True:
			rng = np.random.randint(0,9)
			(y_train, y_validate) = SplitSet(y, rng)
			(x_train, x_validate) = SplitSet(x, rng)

			forestList = []
			y_predictions = np.zeros((y_validate.shape[0], 1), dtype=np.int32)

			for class_idx in range(NUM_CLASSES):
				class_num = class_idx + 1
				forestList.append([])
				for tree_idx in range(NUM_TREES_IN_FOREST):
					print 'Training Tree ', tree_idx, ' of forest ', class_num, ' attribute size ', ATTRIBUTE_SAMPLE_SIZE
					x_sample,y_sample = sample(x_train,y_train,SAMPLE_PROPORTION)
					train_targets = PreProcess(y_sample,class_num)
					if randomsample == True:
						attribute_sample_size = np.random.randint(RANDOMSAMPLELOWER,RANDOMSAMPLEUPPER)
					else:
						attribute_sample_size = ATTRIBUTE_SAMPLE_SIZE
					attribute_set = sample_attributes(range(x.shape[1]),attribute_sample_size)
					tree = Decision_Tree_Learning(x_sample,attribute_set,train_targets)

					forestList[class_idx].append(tree)

			forest_weights = np.zeros((NUM_TREES_IN_FOREST,NUM_CLASSES))
			for forest_idx in range(NUM_CLASSES):
				forest_weights[:,forest_idx] = decision_forest_vote_adaboost(forestList[forest_idx],forest_idx+1,x_validate,y_validate)


			nametrees = 'trees.pkl'#'adaboost_'+str(ATTRIBUTE_SAMPLE_SIZE)+'attributes_'+str(NUM_TREES_IN_FOREST)+'trees.pkl'
			DataDump((forestList, forest_weights), nametrees, 5)
			print 'the trees and their weights are saved in ', nametrees

	else:
			forestList = []
			for class_idx in range(NUM_CLASSES):
				class_num = class_idx + 1
				forestList.append([])
				for tree_idx in range(NUM_TREES_IN_FOREST):
					print 'Training Tree ', tree_idx, ' of forest ', class_num, ' attribute size ', ATTRIBUTE_SAMPLE_SIZE
					x_sample,y_sample = sample(x,y,SAMPLE_PROPORTION)
					train_targets = PreProcess(y_sample,class_num)
					if randomsample == True:
						attribute_sample_size = np.random.randint(RANDOMSAMPLELOWER,RANDOMSAMPLEUPPER)
					else:
						attribute_sample_size = ATTRIBUTE_SAMPLE_SIZE
					attribute_set = sample_attributes(range(x.shape[1]),attribute_sample_size)

					tree = Decision_Tree_Learning(x_sample,attribute_set,train_targets)

					forestList[class_idx].append(tree)

			nametrees = 'trees.pkl' #'rdf_'+str(ATTRIBUTE_SAMPLE_SIZE)+'attributes_'+str(NUM_TREES_IN_FOREST)+'trees.pkl'
			DataDump(forestList, nametrees, 4)
			print 'the trees are saved in ', nametrees

else:
	depth_parameters = [np.zeros(NUM_CLASSES, dtype = np.float64),np.zeros(NUM_CLASSES, dtype = np.float64),np.zeros(NUM_CLASSES, dtype = np.float64),np.zeros(NUM_CLASSES, dtype = np.float64)]

	stacked_depth_parameters = MeasurePerFold(stacked_confusion_matrices_depth, NUM_CLASSES)
	print "###################################################################"
	print "############Average statistics throughout training time############"
	print "###################################################################"
	print "Confusion Matrix for max depth: "
	PrintMatrix(stacked_confusion_matrices_depth)
	print ""
	print "Sample Error: "
	print("Max Depth = %.3f" % (stacked_sample_error_depth/NUM_FOLDS))
	print ""
	print "Stacked Average Measures for max depth: "
	print("Recall = %.3f, Precision = %.3f, F1 = %.3f, CR = %.3f" % (np.sum(stacked_depth_parameters[0,:]/6.0),np.sum(stacked_depth_parameters[1,:]/6.0),np.sum(stacked_depth_parameters[2,:]/6.0),np.sum(stacked_depth_parameters[3,:]/6.0)))
	PrintMatrix(stacked_depth_parameters)
	print ""

	print "adaboost ",adaboost
