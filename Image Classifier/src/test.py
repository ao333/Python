import glob
from src.utils.data_utils import load_pickle
from imageio import imread
from src.fcnet import FullyConnectedNet
import numpy as np
from keras.models import load_model
import pandas as pd
def test_fer_model(img_folder, model='best_model.p'):
    '''
    Given a folder with images, load the images and your best model to
    predict
    the facial expression of each image.
    Args:
    - img_folder: Path to the images to be tested
    Returns:
    - preds: A numpy vector of size N with N being the number of
    images in
    img_folder.
    '''
    preds = None
    ### Start your code here
    preds = []
    model = load_pickle(open(model, "rb" ))
    for filename in sorted(glob.glob(img_folder+'/*')):
        img_data = imread(filename)
        img_data = (img_data.transpose(2, 0, 1).copy()[0]).astype('float64')
        img_data -= model.train_data_mean.reshape(img_data.shape)
        scores = model.loss(np.array([img_data]))
        preds.append(np.argmax(scores))
    ### End of code
    return np.array(preds)

def test_deep_fer_model(img_folder, model='best model.hdf5'):
    preds = []
    # get the mean which the data is substracted from
    model1 = load_pickle(open('best_model.p', "rb" ))
    mean = model1.train_data_mean
    
    # get the best for CNN
    model = load_model(model)
    for filename in sorted(glob.glob(img_folder+'/*')):
        img_data = imread(filename)
        img_data = (img_data.transpose(2, 0, 1).copy()[0]).astype('float64')
        img_data -= mean.reshape(img_data.shape)
        img_data = img_data.reshape(1,48,48,1)
        img_data /= 255
        prediction = model.predict(img_data)
        prediction = np.argmax(prediction, axis=1)
        preds.append(prediction[0])
    
    
    return np.array(preds)











