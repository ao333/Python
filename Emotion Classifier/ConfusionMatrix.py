import numpy as np


def ConfusionMatrix(y_predict,y_true,num_classes):
    confusionMatrix = np.zeros((num_classes,num_classes), dtype=np.int32)
    for index in range(0, len(y_predict)):
        confusionMatrix[int(y_true[index]) - 1, int(y_predict[index]) - 1]  += 1 #Careful because classes start from 1, need to change if start from 0
    return confusionMatrix
