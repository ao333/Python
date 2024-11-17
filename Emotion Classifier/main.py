import scipy.io
import sys
import numpy as np
from SplitSet import *
from PreProcess import *
# from DecisionTree import *
from DecisionTreeMaxDepth import *
from Visualizer import *
from TestTrees import *
from ConfusionMatrix import *
from Measure import *
from CountOccurrence import *
from DeepCopyTree import *
from PrintMatrix import *
from DataDump import *

MIN_DEPTH = 6
NUM_CLASSES = 6
NUM_FOLDS = 10

#-----------------------------------#
# MAIN PROGRAM IS HERE #
#-----------------------------------#

try:
    dataFilename = sys.argv[1]
except IndexError:
    dataFilename = 'cleandata_students.mat'

try:
    ambiguityHandlingStyle = sys.argv[2]
except IndexError:
    ambiguityHandlingStyle = AmbiguousClassificationHandlingStyle.EXTENDED

if not os.path.isfile(dataFilename):
    print 'The data file "' + dataFilename + '" is not found.'
    sys.exit(1)

x,y = parse_data(scipy.io.loadmat(dataFilename))
measurements = []

best_trees_depth = []
best_trees_occ = []
best_trees_depth_min = []
best_depth_cr = 0
best_occ_correct = 0
best_depth_min_cr = 0
occurrences = CountOccurrence(y, NUM_CLASSES)

confusion_matrices_depth = []
confusion_matrices_depth_min = []
confusion_matrices_occ = []

stacked_confusion_matrices_depth = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int32)
stacked_confusion_matrices_depth_min = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int32)
stacked_confusion_matrices_occ = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int32)

stacked_sample_error_depth = 0
stacked_sample_error_depth_min = 0
stacked_sample_error_occ = 0
stacked_sample_error_depth = 0
stacked_sample_error_depth_min = 0
stacked_sample_error_occ = 0

for fold_idx in range(0, NUM_FOLDS): # fold
    fold_num = fold_idx+1
    (y_train_validate, y_test) = SplitSet(y, fold_idx)
    y_predictions = np.zeros((y_test.shape[0], 1), dtype=np.int32)
    (x_train_validate, x_test) = SplitSet(x, fold_idx)
    treeList = []
    occurrences = CountOccurrence(y_train_validate, NUM_CLASSES)


    for tree_class_idx in range(0,NUM_CLASSES):
        class_num = tree_class_idx + 1
        train_targets = PreProcess(y_train_validate,class_num)
        tree = Decision_Tree_Learning(x_train_validate,range(x_train_validate.shape[1]),train_targets,None)
        # if  class_num != 1:
        #     tree = Decision_Tree_Learning(x_train_validate,range(x_train_validate.shape[1]),train_targets,max_depth)
        # else:
        #     tree = Decision_Tree_Learning(x_train_validate,range(x_train_validate.shape[1]),train_targets,10)
        treeList.append(tree)

    predictions_by_depth = TestTreesByDepth(treeList, x_test, ambiguityHandlingStyle)
    predictions_by_min_depth = TestTreesByMinDepth(treeList, x_test, ambiguityHandlingStyle)
    predictions_by_occ = TestTreesByOccurrence(treeList, occurrences, x_test, ambiguityHandlingStyle)

    depth_correct = 0
    depth_min_correct = 0
    occ_correct = 0
    for prediction_idx in range(0, y_test.shape[0]):
        if predictions_by_depth[prediction_idx] == y_test[prediction_idx]:
            depth_correct += 1
        if predictions_by_min_depth[prediction_idx] == y_test[prediction_idx]:
            depth_min_correct += 1
        if predictions_by_occ[prediction_idx] == y_test[prediction_idx]:
            occ_correct += 1

    depth_cr = depth_correct/float(y_test.shape[0])
    depth_min_cr = depth_min_correct/float(y_test.shape[0])
    occ_correct = occ_correct/float(y_test.shape[0])
    # print depth_cr
    stacked_sample_error_depth += 1-(depth_cr)
    stacked_sample_error_depth_min += 1-(depth_min_cr)
    stacked_sample_error_occ += 1-(occ_correct)
    if depth_cr > best_depth_cr:
        best_depth_cr = depth_cr
        best_trees_depth = DeepCopyTreeList(treeList)
    if depth_min_cr > best_depth_min_cr:
        best_depth_min_cr = depth_min_cr
        best_trees_depth_min = DeepCopyTreeList(treeList)
    if occ_correct > best_occ_correct:
        best_occ_correct = occ_correct
        best_trees_occ = DeepCopyTreeList(treeList)

    confusion_matrices_depth.append(ConfusionMatrix(predictions_by_depth,y_test,NUM_CLASSES))
    confusion_matrices_depth_min.append(ConfusionMatrix(predictions_by_min_depth,y_test,NUM_CLASSES))
    confusion_matrices_occ.append(ConfusionMatrix(predictions_by_occ,y_test,NUM_CLASSES))

    stacked_confusion_matrices_depth += ConfusionMatrix(predictions_by_depth,y_test,NUM_CLASSES)
    stacked_confusion_matrices_depth_min += ConfusionMatrix(predictions_by_min_depth,y_test,NUM_CLASSES)
    stacked_confusion_matrices_occ += ConfusionMatrix(predictions_by_occ,y_test,NUM_CLASSES)

depth_parameters = [np.zeros(NUM_CLASSES, dtype = np.float64),np.zeros(NUM_CLASSES, dtype = np.float64),np.zeros(NUM_CLASSES, dtype = np.float64),np.zeros(NUM_CLASSES, dtype = np.float64)]
depth_min_parameters = [np.zeros(NUM_CLASSES, dtype = np.float64),np.zeros(NUM_CLASSES, dtype = np.float64),np.zeros(NUM_CLASSES, dtype = np.float64),np.zeros(NUM_CLASSES, dtype = np.float64)]
occ_parameters = [np.zeros(NUM_CLASSES, dtype = np.float64),np.zeros(NUM_CLASSES, dtype = np.float64),np.zeros(NUM_CLASSES, dtype = np.float64),np.zeros(NUM_CLASSES, dtype = np.float64)]

for matrix_idx in range(0, len(confusion_matrices_depth)):
    depth_parameters += MeasurePerFold(confusion_matrices_depth[matrix_idx], NUM_CLASSES)
    depth_min_parameters += MeasurePerFold(confusion_matrices_depth_min[matrix_idx], NUM_CLASSES)
    occ_parameters += MeasurePerFold(confusion_matrices_occ[matrix_idx], NUM_CLASSES)

depth_parameters /= NUM_FOLDS
depth_min_parameters /= NUM_FOLDS
occ_parameters /= NUM_FOLDS

stacked_depth_parameters = MeasurePerFold(stacked_confusion_matrices_depth, NUM_CLASSES)
stacked_depth_min_parameters = MeasurePerFold(stacked_confusion_matrices_depth_min, NUM_CLASSES)
stacked_occ_parameters = MeasurePerFold(stacked_confusion_matrices_occ, NUM_CLASSES)

selectionMode = [
    'Extended',
    'Random',
    'Shallowest Tree',
    'Deepest Tree',
    'Random Weighted',
    'Random Inverse Weighted'
]

resolutionLabel = ' using ' + selectionMode[int(ambiguityHandlingStyle)] + ' resolution'

for treeIdx in range(len(best_trees_depth)):
    tree = best_trees_depth[treeIdx]
    v = Visualizer('TreeDepth' + str(treeIdx + 1), tree)
    v.build('Best Tree by Depth for Class ' + str(treeIdx) + resolutionLabel)
DataDump(best_trees_depth, 'TreeDepth.pkl', 1)

for treeIdx in range(len(best_trees_depth_min)):
    tree = best_trees_depth_min[treeIdx]
    v = Visualizer('TreeDepthMin' + str(treeIdx + 1), tree)
    v.build('Best Tree by Minimum Depth for Class ' + str(treeIdx) + resolutionLabel)
DataDump(best_trees_depth_min, 'TreeMinDepth.pkl', 2)

for treeIdx in range(len(best_trees_occ)):
    tree = best_trees_occ[treeIdx]
    v = Visualizer('TreeOcc' + str(treeIdx + 1), tree)
    v.build('Best Tree by Occurrence for Class ' + str(treeIdx) + resolutionLabel)
DataDump(best_trees_occ, 'TreeOcc.pkl', 3)

occurrences = CountOccurrence(y, NUM_CLASSES)
depth_prediction = TestTreesByDepth(best_trees_depth, x, ambiguityHandlingStyle)
depth_min_prediction = TestTreesByMinDepth(best_trees_depth_min, x, ambiguityHandlingStyle)
occ_prediction = TestTreesByOccurrence(best_trees_occ, occurrences, x, ambiguityHandlingStyle)

best_depth_CM = ConfusionMatrix(depth_prediction,y,NUM_CLASSES)
best_depth_CM_min = ConfusionMatrix(depth_min_prediction,y,NUM_CLASSES)
best_occ_CM = ConfusionMatrix(occ_prediction,y,NUM_CLASSES)

measure_best_depth = MeasurePerFold(best_depth_CM, NUM_CLASSES)
measure_best_depth_min = MeasurePerFold(best_depth_CM_min, NUM_CLASSES)
measure_best_occ = MeasurePerFold(best_occ_CM, NUM_CLASSES)
print "###################################################################"
print "############Average statistics throughout training time############"
print "###################################################################"
print "Confusion Matrix for max depth: "
PrintMatrix(stacked_confusion_matrices_depth)
print "Confusion Matrix for min depth: "
PrintMatrix(stacked_confusion_matrices_depth_min)
print "Confusion Matrix for occ: "
PrintMatrix(stacked_confusion_matrices_occ)
print ""
print "Sample Error: "
print("Max Depth = %.3f, Min Depth = %.3f, Occ = %.3f" % (stacked_sample_error_depth/NUM_FOLDS,stacked_sample_error_depth_min/NUM_FOLDS, stacked_sample_error_occ/NUM_FOLDS))
print ""

print "Stacked Average Measures for max depth: "
print("Recall = %.3f, Precision = %.3f, F1 = %.3f, CR = %.3f" % (np.sum(stacked_depth_parameters[0,:]/6.0),np.sum(stacked_depth_parameters[1,:]/6.0),np.sum(stacked_depth_parameters[2,:]/6.0),np.sum(stacked_depth_parameters[3,:]/6.0)))
print ""
print "Stack Average Measures for min depth: "
print("Recall = %.3f, Precision = %.3f, F1 = %.3f, CR = %.3f" % (np.sum(stacked_depth_min_parameters[0,:]/6.0),np.sum(stacked_depth_min_parameters[1,:]/6.0),np.sum(stacked_depth_min_parameters[2,:]/6.0),np.sum(stacked_depth_min_parameters[3,:]/6.0)))
print "Stacked Average Measures for occ: "
print("Recall = %.3f, Precision = %.3f, F1 = %.3f, CR = %.3f" % (np.sum(stacked_occ_parameters[0,:]/6.0),np.sum(stacked_occ_parameters[1,:]/6.0),np.sum(stacked_occ_parameters[2,:]/6.0),np.sum(stacked_occ_parameters[3,:]/6.0)))
print ""

print "Average Measures for max depth: "
PrintMatrix(depth_parameters)
print("Recall = %.3f, Precision = %.3f, F1 = %.3f, CR = %.3f" % (np.sum(depth_parameters[0,:]/6.0),np.sum(depth_parameters[1,:]/6.0),np.sum(depth_parameters[2,:]/6.0),np.sum(depth_parameters[3,:]/6.0)))
print ""
print "Average Measures for min depth: "
PrintMatrix(depth_min_parameters)
print("Recall = %.3f, Precision = %.3f, F1 = %.3f, CR = %.3f" % (np.sum(depth_min_parameters[0,:]/6.0),np.sum(depth_min_parameters[1,:]/6.0),np.sum(depth_min_parameters[2,:]/6.0),np.sum(depth_min_parameters[3,:]/6.0)))
print "Average Measures for occ: "
print ""
PrintMatrix(occ_parameters)
print("Recall = %.3f, Precision = %.3f, F1 = %.3f, CR = %.3f" % (np.sum(occ_parameters[0,:]/6.0),np.sum(occ_parameters[1,:]/6.0),np.sum(occ_parameters[2,:]/6.0),np.sum(occ_parameters[3,:]/6.0)))
print ""
