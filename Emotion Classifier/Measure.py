import numpy as np

def MeasurePerFold(matrix, num_classes):
    recall = np.zeros(num_classes, dtype = np.float64)
    precision = np.zeros(num_classes, dtype = np.float64)
    F1 = np.zeros(num_classes, dtype = np.float64)
    CR = np.zeros(num_classes, dtype = np.float64)
    for idx  in range(0, num_classes):
        TP = matrix[idx, idx]
        FP = np.sum(matrix[:, idx]) - TP
        TN = np.trace(matrix) - TP
        FN = np.sum(matrix[idx, :]) - TP
        tprecision = 0
        trecall = 0
        if (TP + TN + FP + FN) == 0:
            CR[idx] = 0
        else:
            CR[idx] = (TP + TN) / (float(TP + TN + FP + FN))
            
        if TP + FN > 0:
            trecall = TP / (float(TP + FN))
            recall[idx] = float(trecall)
        else:
            recall[idx] = 0
            
        if TP + FP > 0:
            tprecision = TP / (float(TP + FP))
            precision[idx] = float(tprecision)
        else:
            precision[idx] = 0
            
        if tprecision + tprecision > 0:
            F1[idx] = 2.0 * (trecall * tprecision) / (trecall + tprecision)
        else:
            F1[idx] =0

    return np.array([recall, precision, F1, CR])

def sampleError (y_predict, y_label):
    delta = 0
    for i in range(0, y_label.shape[0]):
        if y_predict[i] == y_label[i]:
            delta += 0
        else:
            delta +=1
    return (delta/float(y_label.shape[0]))
