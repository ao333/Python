import numpy as np
import random as rnd
from GetTreeHeight import *

class AmbiguousClassificationHandlingStyle:
    EXTENDED = 0
    RANDOM = 1
    SHALLOWEST = 2
    DEEPEST = 3
    RANDOMWEIGHTED = 4
    RANDOMINVWEIGHTED = 5

def RunRowByTreeByDepth(tree, row):
    def RunRowByTreeByDepthRecursive(treeNode, depth):
        if len(treeNode.kids) == 0:
            return (treeNode.label, depth)
        path = row[treeNode.op]
        for child in treeNode.kids:
            if child.sign == path:
                return RunRowByTreeByDepthRecursive(child, depth + 1)
    return RunRowByTreeByDepthRecursive(tree, 0)

def RunRowByTree(tree, row):
    return RunRowByTreeByDepth(tree, row)[0]

def GetIndexOfShallowestTrees(trees):
    minDepth = GetTreeHeight(trees[0])
    resultIndex = 0
    for treeIdx in range(1, len(trees)):
        currentTreeHeight = GetTreeHeight(trees[treeIdx])
        if minDepth > currentTreeHeight:
            minDepth = currentTreeHeight
            resultIndex = treeIdx
    return resultIndex

def GetIndexOfDeepestTrees(trees):
    maxDepth = GetTreeHeight(trees[0])
    resultIndex = 0
    for treeIdx in range(1, len(trees)):
        currentTreeHeight = GetTreeHeight(trees[treeIdx])
        if maxDepth < currentTreeHeight:
            maxDepth = currentTreeHeight
            resultIndex = treeIdx
    return resultIndex

def translate(value, leftMin, leftMax, rightMin, rightMax):
    # Figure out how 'wide' each range is
    leftSpan = leftMax - leftMin
    rightSpan = rightMax - rightMin

    # Convert the left range into a 0-1 range (float)
    valueScaled = float(value - leftMin) / float(leftSpan)

    # Convert the 0-1 range into a value in the right range.
    return rightMin + (valueScaled * rightSpan)

def handleAmbiguity(selectedClassIdx, handlingStyle, deepestTreeIndex, shallowestTreeIndex, num_classes):
    if handlingStyle == AmbiguousClassificationHandlingStyle.EXTENDED:
        selectedClassIdx = selectedClassIdx - 1

    if int(handlingStyle) == int(AmbiguousClassificationHandlingStyle.RANDOM):

        selected= rnd.randint(0, num_classes - 1)
        return selected

    if int(handlingStyle) == int(AmbiguousClassificationHandlingStyle.SHALLOWEST):
        index = np.argmin(selectedClassIdx)
        return index

    if int(handlingStyle) == int(AmbiguousClassificationHandlingStyle.DEEPEST):
        index = np.argmax(selectedClassIdx)
        return index

    if int(handlingStyle) == int(AmbiguousClassificationHandlingStyle.RANDOMWEIGHTED):

        occurrences = np.array([88, 187, 187,209,110,220])

        elements = range(num_classes)
        probabilities = occurrences/float(sum(occurrences))
        selected = np.random.choice(elements, 1, p=probabilities)[0]
        return selected

    if int(handlingStyle) == int(AmbiguousClassificationHandlingStyle.RANDOMINVWEIGHTED):

        occurrences = np.array([88, 187, 187,209,110,220])
        inverse_occ= np.zeros(6)
        for i in range(num_classes):
            inverse_occ[i] = translate(occurrences[i],88,220,220,88)
        elements = range(num_classes)
        probabilities = inverse_occ/float(sum(inverse_occ))
        selected = np.random.choice(elements, 1, p=probabilities)[0]
        return selected

def no_class(binary_targets):
    for i in binary_targets:
        if i != -1:
            return False
    return True
def no_class_2(binary_targets):
    for i in binary_targets:
        if i != 10000:
            return False
    return True
def TestTreesByDepth(trees, x, handlingStyle):

    num_classes = len(trees)
    deepestTreeIndex = GetIndexOfDeepestTrees(trees)
    shallowestTreeIndex = GetIndexOfShallowestTrees(trees)
    result = np.zeros((x.shape[0], 1), dtype=np.int32)

    for rowIdx in range(0, x.shape[0]):
        row = x[rowIdx, :]
        depthRow = np.zeros(num_classes, dtype=np.int32)
        negative = np.zeros(num_classes, dtype=np.int32)
        for classIdx in range(0, num_classes):
            treeResult, depth = RunRowByTreeByDepth(trees[classIdx], row)

            if treeResult == 0:
                depthRow[classIdx] = -1
                negative[classIdx] = depth
            else:
                # we store the depth in the depth row
                depthRow[classIdx] = depth
                negative[classIdx] = -1

        if no_class(depthRow)==False:
            selectedClassIdx = np.argmax(depthRow)

        # print selectedClassIdx
        else:
            # selectedClassIdx = np.argmax(negative)
            # print negative
            # print selectedClassIdx
            selectedClassIdx = handleAmbiguity(negative, handlingStyle, deepestTreeIndex, shallowestTreeIndex, num_classes)
        result[rowIdx, 0] = selectedClassIdx + 1
        # print result
    return result

def TestTreesByMinDepth(trees, x, handlingStyle):

    num_classes = len(trees)
    deepestTreeIndex = GetIndexOfDeepestTrees(trees)
    shallowestTreeIndex = GetIndexOfShallowestTrees(trees)
    result = np.zeros((x.shape[0], 1), dtype=np.int32)

    for rowIdx in range(0, x.shape[0]):
        row = x[rowIdx, :]
        depthRow = np.zeros(num_classes, dtype=np.int32)
        negative = np.zeros(num_classes, dtype=np.int32)
        for classIdx in range(0, num_classes):
            treeResult, depth = RunRowByTreeByDepth(trees[classIdx], row)
            if treeResult == 0:
                depthRow[classIdx] = 10000
                negative[classIdx] = depth
            else:
                # we store the depth in the depth row
                depthRow[classIdx] = depth
                negative[classIdx] = -1
        if no_class_2(depthRow) == False:
            selectedClassIdx = np.argmin(depthRow)
        else:
            selectedClassIdx = np.argmax(negative)

            selectedClassIdx = handleAmbiguity(negative, handlingStyle, deepestTreeIndex, shallowestTreeIndex, num_classes)
        result[rowIdx, 0] = selectedClassIdx + 1
    return result

def TestTreesByOccurrence(trees, occurrences, x, handlingStyle):
    num_classes = len(trees)
    deepestTreeIndex = GetIndexOfDeepestTrees(trees)
    shallowestTreeIndex = GetIndexOfShallowestTrees(trees)
    result = np.zeros((x.shape[0], 1), dtype=np.int32)

    for rowIdx in range(0, x.shape[0]):
        row = x[rowIdx, :]
        depthRow = np.zeros(num_classes, dtype=np.int32)
        negative = np.zeros(num_classes, dtype=np.int32)
        for classIdx in range(0, num_classes):
            treeResult,depth = RunRowByTreeByDepth(trees[classIdx], row)
            if treeResult == 0:
                depthRow[classIdx] = -1
                negative[classIdx] = depth
            else:
                depthRow[classIdx] = occurrences[classIdx]
                negative[classIdx] = 10000
        if no_class(depthRow) == False:
            selectedClassIdx = np.argmax(depthRow)
        else:
            # selectedClassIdx = np.argmax(negative)
            # elements = range(num_classes)
            # probabilities = occurrences/float(sum(occurrences))
            # selectedClassIdx = np.random.choice(elements, 1, p=probabilities)[0]
            selectedClassIdx = handleAmbiguity(negative, handlingStyle, deepestTreeIndex, shallowestTreeIndex, num_classes)
        result[rowIdx, 0] = selectedClassIdx + 1
    return result
