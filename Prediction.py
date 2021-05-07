import os
import shutil
import random
import keras
import tensorflow as tf

from keras.preprocessing.image import ImageDataGenerator,load_img
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.models import load_model
from keras.utils.generic_utils import get_custom_objects
import keras.losses
import matplotlib.pyplot as plt


def load_and_predict():
    model = keras.models.load_model('model1_catsVSdogs_10epoch.h5')

    test_generator = ImageDataGenerator(rescale=1. / 255)

    test_iterator = test_generator.flow_from_directory(
        './input_test',
        target_size=(150, 150),
        shuffle=False,
        class_mode='binary',
        batch_size=1)

    ids = []
    for filename in test_iterator.filenames:
        ids.append(int(filename.split('\\')[1].split('.')[0]))

    predict_result = model.predict(test_iterator, steps=len(test_iterator.filenames))
    predictions = []
    for index, prediction in enumerate(predict_result):
        predictions.append([ids[index], prediction[0]])
    predictions.sort()
    
    return predictions


predictions = load_and_predict()
df = pd.DataFrame(data=predictions, index=range(1, 12501), columns=['id', 'label'])
df = df.set_index(['id'])
df.to_csv('submission.csv')