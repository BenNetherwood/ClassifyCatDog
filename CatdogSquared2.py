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


def distribute_train_validation_split(validation_size=0.2):

    all_images = os.listdir('./dogs-vs-cats/train/')
    random.shuffle(all_images)

    all_dogs = list(filter(lambda image: 'dog' in image, all_images))
    all_cats = list(filter(lambda image: 'cat' in image, all_images))

    index_to_split = int(len(all_dogs) - len(all_dogs) * validation_size)
    training_dogs = all_dogs[:index_to_split]
    validation_dogs = all_dogs[index_to_split:]
    training_cats = all_cats[:index_to_split]
    validation_cats = all_cats[index_to_split:]

    shutil.rmtree('./input_for_model')
    os.makedirs('./input_for_model/train/dogs/', exist_ok=True)
    os.makedirs('./input_for_model/train/cats/', exist_ok=True)
    os.makedirs('./input_for_model/validation/dogs/', exist_ok=True)
    os.makedirs('./input_for_model/validation/cats/', exist_ok=True)

    copy_images_to_dir(training_dogs, './input_for_model/train/dogs')
    copy_images_to_dir(validation_dogs, './input_for_model/validation/dogs')
    copy_images_to_dir(training_cats, './input_for_model/train/cats')
    copy_images_to_dir(validation_cats, './input_for_model/validation/cats')

def copy_images_to_dir(images_to_copy, destination):
   for image in images_to_copy:
        shutil.copyfile(f'./dogs-vs-cats/train/{image}', f'{destination}/{image}')
        
distribute_train_validation_split(0.25)
# gen artifical data through augmentation 
train_imagedatagenerator = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')



validation_imagedatagenerator = ImageDataGenerator(rescale=1/255.0)

train_iterator = train_imagedatagenerator.flow_from_directory(
    './input_for_model/train',
    target_size=(150, 150),
    batch_size=200,
    class_mode='binary')

validation_iterator = validation_imagedatagenerator.flow_from_directory(
    './input_for_model/validation',
    target_size=(150, 150),
    batch_size=50,
    class_mode='binary')



model = keras.Sequential([
    keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    keras.layers.MaxPool2D((2, 2)),
    keras.layers.Conv2D(32, (3, 3), activation='relu'),
    keras.layers.MaxPool2D((2, 2)),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPool2D((2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(512, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='BinaryCrossentropy', metrics=['accuracy'])
model.summary()




# train model 

history = model.fit(train_iterator,
                    validation_data=validation_iterator,
                    steps_per_epoch=90,
                    epochs=100,
                    validation_steps=100)


def plot_result(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    epochs = range(len(acc))

    plt.plot(epochs, acc, 'b', label='Training accuracy')
    plt.plot(epochs, val_acc, 'r', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.figure()

    loss = history.history['loss']
    val_loss = history.history['val_loss']
    plt.plot(epochs, loss, 'b', label='Training Loss')
    plt.plot(epochs, val_loss, 'r', label='Validation Loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()

plot_result(history)







