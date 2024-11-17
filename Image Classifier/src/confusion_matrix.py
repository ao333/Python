import itertools
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from src.utils.data_utils import get_FER2013_data
from keras.models import load_model

#calculate the F1 measure per class
def calFAlpha(confusion_mat):
        FAlpha = []
        for index in range(7):
            tp = confusion_mat[index][index]
            fn = sum(confusion_mat[index]) - tp
            fp = sum([i[index] for i in confusion_mat]) - tp
            if tp == 0:
                FAlpha.append(0)
            else:
                recall_rate = tp/float(tp+fn)
                precision_rate = tp/float(tp+fp)
                FAlpha.append(2*precision_rate*recall_rate/(precision_rate+recall_rate))
        return FAlpha
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function plots the confusion matrix.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

model = load_model('models/best model.hdf5')
data = get_FER2013_data(num_test = 3589)
x_test = data['X_test'] / 255
x_test = x_test.reshape(3589,48,48,1)
y_test = data['y_test'] 

y_pre = model.predict(x_test) 
y_pre = np.argmax(y_pre, axis=1)
confusion_matrix = confusion_matrix(y_test, y_pre)
accuracy = np.trace(confusion_matrix) / np.sum(confusion_matrix)

class_name = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

plot_confusion_matrix(confusion_matrix, class_name,
                          normalize=False,
                          title='Confusion matrix',)
plt.savefig('confusion matrix')

FAlpha = []
FAlpha = calFAlpha(confusion_matrix)
FAlpha = np.around(FAlpha, decimals=2)

plt.show()
