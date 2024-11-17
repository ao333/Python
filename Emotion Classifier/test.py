import scipy.io
import sys
import numpy as np
from SplitSet import *
from PreProcess import *
from DecisionTree import *
from Visualizer import *
from TestTrees import *
from ConfusionMatrix import *
from Measure import *
from CountOccurrence import *
from DeepCopyTree import *
from PrintMatrix import *
from DataDump import *
from rdffunctions import *
import os.path

NUM_CLASSES = 6

try:
    dataFilename = sys.argv[1]
except IndexError:
    dataFilename = 'cleandata_students.mat'
try:
    treeFilename = sys.argv[2]
except IndexError:
    treeFilename = 'TreeDepth.pkl'
ambiguityHandlingStyle = AmbiguousClassificationHandlingStyle.RANDOM

if not os.path.isfile(dataFilename):
    print 'The data file "' + dataFilename + '" is not found.'
    sys.exit(1)

if not os.path.isfile(treeFilename):
    print 'The tree file "' + treeFilename + '" is not found.'
    sys.exit(1)

print "Loading data file ", dataFilename
x,y = parse_data(scipy.io.loadmat(dataFilename))
print "Data file loaded"
print ""

print "Loading trees file ", treeFilename
(testMethod, trees) = DataLoad(treeFilename)
print "Trees file loaded"
print ""
if testMethod == 2:
    print "Testing Tree by Min Depth"
    prediction = TestTreesByMinDepth(trees, x, ambiguityHandlingStyle)
elif testMethod == 3:
    print "Testing Tree by Occurrence"
    occurrences = CountOccurrence(y, NUM_CLASSES)
    print "Occurrence Info: ", occurrences
    prediction = TestTreesByOccurrence(trees, occurrences, x, ambiguityHandlingStyle)
elif testMethod == 4:
    print "Testing with RDF"
    vote_block = np.zeros((x.shape[0], NUM_CLASSES))
    for forest_idx in range(NUM_CLASSES):
        choices = decision_forest_vote(trees[forest_idx], x)
        vote_block[:,forest_idx] = choices

    prediction = np.zeros((x.shape[0], 1))

    for vote_idx in range(vote_block.shape[0]):
        prediction[vote_idx] = int(np.argmax(vote_block[vote_idx]))+1
elif testMethod == 5:
    print "Testing with RDF with AdaBoost"
    (forest, weights) = trees
    vote_block = np.zeros((x.shape[0], NUM_CLASSES))
    for forest_idx in range(NUM_CLASSES):
        choices = decision_forest_vote_weighted(forest[forest_idx], weights[:,forest_idx], x)
        vote_block[:,forest_idx] = choices

    prediction = np.zeros((x.shape[0], 1))

    for vote_idx in range(vote_block.shape[0]):
        prediction[vote_idx] = int(np.argmax(vote_block[vote_idx]))+1
else:
    print "Testing Tree by Max Depth"
    prediction = TestTreesByDepth(trees, x, ambiguityHandlingStyle)
print "Test Done"
print ""

cm = ConfusionMatrix(prediction, y, NUM_CLASSES)
error = sampleError(prediction, y)
measures = MeasurePerFold(cm, NUM_CLASSES)

print "Confusion Matrix"
PrintMatrix(cm)
print ""

PrintMatrix(measures)
print "Recall = %.3f, Precision = %.3f, F1 = %.3f, CR = %.3f" % (np.sum(measures[0,:]/6.0),np.sum(measures[1,:]/6.0),np.sum(measures[2,:]/6.0),np.sum(measures[3,:]/6.0))
print ""


print "Error Rate: %.3f" % error
print ""

def save_predictions(y):
    file_object = open('predictions.txt','w')
    for prediction in y:
        file_object.write(str(int(prediction[0])))
        file_object.write('\n')
    file_object.close()

save_predictions(prediction)
print 'prediction saved to predictions.txt'
