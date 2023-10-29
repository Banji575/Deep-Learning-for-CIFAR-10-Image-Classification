import tensorflow as tf
from tensorflow.keras import datasets, layers, models, optimizers
import numpy as np
IMG_CHANNELS = 3
IMG_ROWS = 32
IMG_COLS = 32

EPOCHS = 40
BATCH_SIZE = 64
VERBOSE = 1
OPTIMIZER = tf.keras.optimizers.RMSprop()
VALIDATION_SPLIT = .3
NUM_CLASSES = 10

(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

mean = np.mean(x_train, axis = (0,1,2,3))
std = np.std(x_train, axis = (0,1,2,3))
x_train = (x_train-mean)/(std + 1e-7)
x_test = (x_test - mean)/(std+1e-7)

y_train = tf.keras.utils.to_categorical(y_train, NUM_CLASSES)
y_test = tf.keras.utils.to_categorical(y_test, NUM_CLASSES)



def build(input_shape, classes):
    model = models.Sequential()
    #1block
    model.add(layers.Convolution2D(32, (3,3), padding = 'same', activation = 'relu', input_shape = input_shape))
    model.add(layers.BatchNormalization())
    model.add(layers.Convolution2D(32, (3,3),padding = 'same', activation = 'relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size = (2,2)))
    model.add(layers.Dropout(.2))
    
    #2block
    
    model.add(layers.Convolution2D(64, (3,3),padding = 'same', activation = 'relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Convolution2D(64, (3,3),padding = 'same', activation = 'relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size = (2,2)))
    model.add(layers.Dropout(.3))
    
    #3block
    model.add(layers.Convolution2D(128, (3,3),padding = 'same', activation = 'relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Convolution2D(128, (3,3),padding = 'same', activation = 'relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size = (2,2)))
    model.add(layers.Dropout(.4))
    
    #dense
    
    model.add(layers.Flatten())
    model.add(layers.Dense(classes, activation = 'softmax'))
    return model

model = build((32,32,3), NUM_CLASSES)
model.summary()
model.compile(loss = 'categorical_crossentropy', optimizer = OPTIMIZER, metrics = ['accuracy'])

model.fit(x_train, y_train, batch_size = BATCH_SIZE, epochs = EPOCHS, validation_split = VALIDATION_SPLIT,
         verbose = VERBOSE)  
score = model.evaluate(x_test, y_test, batch_size = BATCH_SIZE, verbose = VERBOSE)
